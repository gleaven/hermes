/* ============================================================
   Stage 3: AUGMENTATION — Albumentations Data Augmentation
   ============================================================ */
const AugmentationStage = (() => {
    const AUGMENTATIONS = [
        { key: 'rain', name: 'Rain' },
        { key: 'fog', name: 'Fog' },
        { key: 'blur', name: 'Motion Blur' },
        { key: 'flip_h', name: 'H-Flip' },
        { key: 'rotate', name: 'Rotate 90' },
        { key: 'brightness', name: 'Brightness' },
        { key: 'color_jitter', name: 'Color Jitter' },
        { key: 'noise', name: 'Noise' },
        { key: 'cutout', name: 'Cutout' },
    ];

    let selectedAugs = new Set(['flip_h', 'brightness', 'blur']);
    let multiplier = 3;
    let previewLoaded = false;

    function getEl(id) { return document.getElementById(id); }

    function init() {
        renderToggles();
        setupMultiplier();

        getEl('btn-generate')?.addEventListener('click', generateDataset);

        EventBus.on('stage:entered', stage => {
            if (stage === 3) onEnter();
        });
    }

    function renderToggles() {
        const container = getEl('aug-toggles');
        container.innerHTML = '';
        AUGMENTATIONS.forEach(aug => {
            const toggle = document.createElement('div');
            toggle.className = `aug-toggle ${selectedAugs.has(aug.key) ? 'active' : ''}`;
            toggle.innerHTML = `
                <div class="aug-checkbox"></div>
                <span class="aug-toggle-name">${aug.name}</span>
            `;
            toggle.addEventListener('click', () => {
                if (selectedAugs.has(aug.key)) {
                    selectedAugs.delete(aug.key);
                    toggle.classList.remove('active');
                } else {
                    selectedAugs.add(aug.key);
                    toggle.classList.add('active');
                }
                pipelineState.augConfig.augmentations = [...selectedAugs];
                updateCounter();
                loadPreview();
            });
            container.appendChild(toggle);
        });
    }

    function setupMultiplier() {
        document.querySelectorAll('.mult-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                multiplier = parseInt(btn.dataset.mult);
                pipelineState.augConfig.multiplier = multiplier;
                document.querySelectorAll('.mult-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                updateCounter();
            });
        });
    }

    function updateCounter() {
        const ds = pipelineState.selectedDataset;
        const from = ds?.image_count || 0;
        const to = from * multiplier;
        getEl('aug-counter').querySelector('.count-from').textContent = from;
        getEl('aug-counter').querySelector('.count-to').textContent = to;
    }

    async function onEnter() {
        updateCounter();
        if (!previewLoaded) loadPreview();
    }

    async function loadPreview() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        // Get first image
        const data = await api(`api/datasets/${ds.id}/images?per_page=1`);
        if (!data.images?.length) return;

        const imageName = data.images[0].name;
        const augList = [...selectedAugs];

        if (augList.length === 0) {
            renderPreviewGrid(null, []);
            return;
        }

        try {
            const result = await api('api/augment/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: ds.id,
                    image_name: imageName,
                    augmentations: augList,
                }),
            });

            renderPreviewGrid(result.original, result.variants);
            previewLoaded = true;
        } catch (e) { /* error shown */ }
    }

    function renderPreviewGrid(original, variants) {
        const grid = getEl('aug-preview-grid');
        grid.innerHTML = '';

        // Original center card
        const origCard = document.createElement('div');
        origCard.className = 'aug-card aug-center';
        if (original) {
            origCard.innerHTML = `
                <img src="data:image/jpeg;base64,${original}" alt="Original">
                <div class="aug-label">ORIGINAL</div>
            `;
        } else {
            origCard.innerHTML = `<div class="aug-label">ORIGINAL</div>`;
        }
        grid.appendChild(origCard);

        // Variant cards
        variants.forEach(v => {
            const card = document.createElement('div');
            card.className = 'aug-card';
            card.innerHTML = `
                <img src="data:image/jpeg;base64,${v.image}" alt="${v.display_name}">
                <div class="aug-label">${v.display_name.toUpperCase()}</div>
            `;
            grid.appendChild(card);
        });
    }

    async function generateDataset() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        const augList = [...selectedAugs];
        if (augList.length === 0) {
            showToast('Select at least one augmentation', 'error');
            return;
        }

        const btn = getEl('btn-generate');
        btn.disabled = true;
        btn.textContent = 'GENERATING...';
        showToast('Applying augmentations to dataset...', 'info');

        try {
            const result = await api('api/augment/apply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: ds.id,
                    augmentations: augList,
                    multiplier: multiplier,
                }),
            });

            showToast(
                `Dataset expanded: ${result.original_count} → ${result.final_count} images`,
                'success'
            );

            // Update dataset info
            pipelineState.selectedDataset.image_count = result.final_count;
            document.getElementById('metric-images').textContent = result.final_count;
            document.getElementById('metric-augmented').textContent = `${result.multiplier}x`;

            unlockStage(4);
        } catch (e) { /* error shown */ } finally {
            btn.disabled = false;
            btn.textContent = 'GENERATE DATASET';
        }
    }

    return { init };
})();
