/* ============================================================
   Stage 1: DATA COLLECTION — Dataset Selection & Upload
   ============================================================ */
const CollectionStage = (() => {
    const grid = () => document.getElementById('dataset-grid');
    const uploadZone = () => document.getElementById('upload-zone');
    const fileInput = () => document.getElementById('file-input');
    const btnNext = () => document.getElementById('btn-next-1');

    async function loadDatasets() {
        const datasets = await api('api/datasets');
        renderGrid(datasets);

        // Auto-select satellite dataset if just created from Stage 0
        if (pipelineState._satelliteDatasetId) {
            const satDs = datasets.find(d => d.id === pipelineState._satelliteDatasetId);
            if (satDs) {
                selectDataset(satDs);
                pipelineState._satelliteDatasetId = null;
            }
        }
    }

    function renderGrid(datasets) {
        const g = grid();
        g.innerHTML = '';

        datasets.forEach(ds => {
            const card = document.createElement('div');
            card.className = 'dataset-card';
            if (pipelineState.selectedDataset?.id === ds.id) card.classList.add('selected');

            const classCount = (ds.classes || []).length;
            const badge = ds.source === 'sample' ? '<span class="dataset-badge">SAMPLE</span>' : '';
            const domainBadge = ds.domain ? `<span class="dataset-domain">${ds.domain}</span>` : '';
            const desc = ds.description ? `<div class="dataset-desc">${ds.description}</div>` : '';

            // Action buttons for user datasets (not samples)
            const actions = ds.source !== 'sample' ? `
                <div class="dataset-actions">
                    <button class="ds-action-btn ds-rename-btn" title="Rename" data-id="${ds.id}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                    <button class="ds-action-btn ds-delete-btn" title="Delete" data-id="${ds.id}">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-2 14H7L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
                    </button>
                </div>
            ` : '';

            card.innerHTML = `
                <div class="dataset-mosaic" id="mosaic-${ds.id}"></div>
                <div class="dataset-info">
                    ${domainBadge}
                    <div class="dataset-name">${ds.name || ds.id}</div>
                    <div class="dataset-meta">${ds.image_count} images &bull; ${classCount} classes</div>
                    ${desc}
                    ${badge}
                </div>
                ${actions}
            `;

            card.addEventListener('click', (e) => {
                // Don't select when clicking action buttons
                if (e.target.closest('.ds-action-btn')) return;
                selectDataset(ds);
            });
            g.appendChild(card);

            // Wire action buttons
            const renameBtn = card.querySelector('.ds-rename-btn');
            const deleteBtn = card.querySelector('.ds-delete-btn');
            renameBtn?.addEventListener('click', (e) => { e.stopPropagation(); renameDataset(ds); });
            deleteBtn?.addEventListener('click', (e) => { e.stopPropagation(); deleteDataset(ds); });

            // Load thumbnail mosaic
            loadMosaic(ds);
        });
    }

    async function loadMosaic(ds) {
        const mosaic = document.getElementById(`mosaic-${ds.id}`);
        if (!mosaic) return;
        try {
            const data = await api(`api/datasets/${ds.id}/images?per_page=4`);
            const images = data.images.slice(0, 4);
            mosaic.innerHTML = images.map(img =>
                `<img src="api/datasets/${ds.id}/images/${encodeURIComponent(img.name)}" alt="${img.name}" onerror="this.style.display='none'">`
            ).join('');

            for (let i = images.length; i < 4; i++) {
                mosaic.innerHTML += `<div></div>`;
            }
        } catch (e) {
            console.warn(`Mosaic load failed for ${ds.id}:`, e);
        }
    }

    function selectDataset(ds) {
        pipelineState.selectedDataset = ds;
        pipelineState.classes = ds.classes || [];
        updateMetrics();
        unlockStage(2);
        btnNext().disabled = false;

        const hasLabels = (ds.labeled_count || 0) > 0;
        const hasClasses = (ds.classes || []).length > 0;
        if (hasLabels && hasClasses) {
            unlockStage(3);
            unlockStage(4);
            unlockStage(5);
        }

        document.querySelectorAll('.dataset-card').forEach(c => c.classList.remove('selected'));
        event?.target?.closest('.dataset-card')?.classList.add('selected');

        showToast(`Dataset "${ds.name}" selected`, 'success');
    }

    async function renameDataset(ds) {
        const newName = prompt('Rename dataset:', ds.name || ds.id);
        if (!newName || newName.trim() === ds.name) return;

        try {
            await api(`api/datasets/${ds.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName.trim() }),
            });
            showToast(`Renamed to "${newName.trim()}"`, 'success');
            if (pipelineState.selectedDataset?.id === ds.id) {
                pipelineState.selectedDataset.name = newName.trim();
                updateMetrics();
            }
            await loadDatasets();
        } catch (e) { /* error shown by api() */ }
    }

    async function deleteDataset(ds) {
        if (!confirm(`Delete dataset "${ds.name || ds.id}"?\n\nThis will permanently remove all images and labels.`)) return;

        try {
            await api(`api/datasets/${ds.id}`, { method: 'DELETE' });
            showToast(`Dataset "${ds.name}" deleted`, 'success');
            if (pipelineState.selectedDataset?.id === ds.id) {
                pipelineState.selectedDataset = null;
                btnNext().disabled = true;
                updateMetrics();
            }
            await loadDatasets();
        } catch (e) { /* error shown by api() */ }
    }

    function setupUpload() {
        const zone = uploadZone();
        const input = fileInput();

        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
        });
        input.addEventListener('change', () => {
            if (input.files.length) uploadFile(input.files[0]);
        });
    }

    async function uploadFile(file) {
        showToast(`Uploading ${file.name}...`, 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const result = await api('api/upload', {
                method: 'POST',
                body: formData,
            });
            showToast(`Dataset created: ${result.image_count} images`, 'success');
            await loadDatasets();
        } catch (e) {
            // Error already shown by api()
        }
    }

    return {
        init() {
            setupUpload();
            EventBus.on('stage:entered', stage => {
                if (stage === 1) loadDatasets();
            });
            loadDatasets();
        }
    };
})();
