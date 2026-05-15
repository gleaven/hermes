/* ============================================================
   Stage 6: MODEL TESTING — Inference + Supervision Visualization
   ============================================================ */
const AssessmentStage = (() => {
    let currentStyle = 'box_corner';
    let confidence = 0.25;
    let selectedImage = null;
    let selectedSource = null; // 'dataset' or 'upload'

    function getEl(id) { return document.getElementById(id); }

    function init() {
        // Style selector
        document.querySelectorAll('.style-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentStyle = btn.dataset.style;
                if (selectedImage) runDetection();
            });
        });

        // Confidence slider
        const slider = getEl('confidence-slider');
        slider?.addEventListener('input', () => {
            confidence = parseInt(slider.value) / 100;
            getEl('confidence-value').textContent = confidence.toFixed(2);
        });
        slider?.addEventListener('change', () => {
            if (selectedImage) runDetection();
        });

        getEl('btn-run-detect')?.addEventListener('click', runDetection);
        getEl('btn-export-model')?.addEventListener('click', exportModel);

        // Test image upload
        const uploadInput = getEl('upload-test-input');
        uploadInput?.addEventListener('change', handleUpload);

        EventBus.on('stage:entered', stage => {
            if (stage === 6) onEnter();
        });
    }

    async function onEnter() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        // Load dataset image thumbnails
        const data = await api(`api/datasets/${ds.id}/images?per_page=20`);
        renderDatasetNav(ds, data.images);

        // Load any previously uploaded test images
        loadUploadedImages();

        // Auto-select first dataset image
        if (data.images.length > 0 && !selectedImage) {
            selectDatasetImage(ds, data.images[0]);
        }

        // Update export button state
        updateExportButton();
    }

    function renderDatasetNav(ds, images) {
        const nav = getEl('detect-image-nav');
        nav.innerHTML = '';
        images.forEach(img => {
            const thumb = document.createElement('div');
            thumb.className = `nav-thumb ${selectedSource === 'dataset' && selectedImage === img.name ? 'active' : ''}`;
            thumb.innerHTML = `<img src="api/datasets/${ds.id}/images/${img.name}" loading="lazy" alt="">`;
            thumb.addEventListener('click', () => selectDatasetImage(ds, img));
            nav.appendChild(thumb);
        });
    }

    function selectDatasetImage(ds, img) {
        selectedImage = img.name;
        selectedSource = 'dataset';

        // Clear all active states
        document.querySelectorAll('#detect-image-nav .nav-thumb, #detect-uploaded-nav .nav-thumb')
            .forEach(t => t.classList.remove('active'));
        // Re-find and activate the clicked thumb
        const navThumbs = document.querySelectorAll('#detect-image-nav .nav-thumb');
        const images = Array.from(navThumbs);
        const idx = Array.from(document.querySelectorAll('#detect-image-nav .nav-thumb img'))
            .findIndex(i => i.src.includes(encodeURIComponent(img.name)) || i.src.endsWith(img.name));
        if (idx >= 0) images[idx].classList.add('active');

        // Show original image
        const resultImg = getEl('detect-result-img');
        resultImg.src = `api/datasets/${ds.id}/images/${img.name}`;
        resultImg.style.display = 'block';
        getEl('detect-placeholder').style.display = 'none';
    }

    async function handleUpload(e) {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                await api('api/test-images/upload', {
                    method: 'POST',
                    body: formData,
                });
            } catch (err) {
                showToast(`Failed to upload ${file.name}`, 'error');
            }
        }

        showToast(`Uploaded ${files.length} test image${files.length > 1 ? 's' : ''}`, 'success');
        e.target.value = ''; // reset file input
        loadUploadedImages();
    }

    async function loadUploadedImages() {
        try {
            const data = await api('api/test-images');
            renderUploadedNav(data.images);
        } catch (e) { /* ignore */ }
    }

    function renderUploadedNav(images) {
        const nav = getEl('detect-uploaded-nav');
        nav.innerHTML = '';
        images.forEach(name => {
            const thumb = document.createElement('div');
            thumb.className = `nav-thumb ${selectedSource === 'upload' && selectedImage === name ? 'active' : ''}`;
            thumb.innerHTML = `<img src="api/test-images/${encodeURIComponent(name)}" loading="lazy" alt="">`;
            thumb.addEventListener('click', () => selectUploadedImage(name));
            nav.appendChild(thumb);
        });
    }

    function selectUploadedImage(name) {
        selectedImage = name;
        selectedSource = 'upload';

        // Clear all active states
        document.querySelectorAll('#detect-image-nav .nav-thumb, #detect-uploaded-nav .nav-thumb')
            .forEach(t => t.classList.remove('active'));
        const thumbs = document.querySelectorAll('#detect-uploaded-nav .nav-thumb img');
        thumbs.forEach(img => {
            if (img.src.includes(encodeURIComponent(name))) {
                img.closest('.nav-thumb').classList.add('active');
            }
        });

        // Show original image
        const resultImg = getEl('detect-result-img');
        resultImg.src = `api/test-images/${encodeURIComponent(name)}`;
        resultImg.style.display = 'block';
        getEl('detect-placeholder').style.display = 'none';
    }

    async function runDetection() {
        if (!selectedImage || !selectedSource) {
            showToast('Select an image first', 'error');
            return;
        }

        const btn = getEl('btn-run-detect');
        btn.disabled = true;
        btn.textContent = 'DETECTING...';

        try {
            let endpoint, payload;
            if (selectedSource === 'upload') {
                endpoint = 'api/detect/annotate-upload';
                payload = {
                    image_name: selectedImage,
                    model_path: pipelineState.trainedModelPath,
                    confidence: confidence,
                    style: currentStyle,
                };
            } else {
                const ds = pipelineState.selectedDataset;
                endpoint = 'api/detect/annotate';
                payload = {
                    dataset_id: ds.id,
                    image_name: selectedImage,
                    model_path: pipelineState.trainedModelPath,
                    confidence: confidence,
                    style: currentStyle,
                };
            }

            const result = await api(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            // Display annotated image
            const resultImg = getEl('detect-result-img');
            resultImg.src = `data:image/jpeg;base64,${result.image}`;
            resultImg.style.display = 'block';
            getEl('detect-placeholder').style.display = 'none';

            // Update stats
            getEl('detect-count').textContent = result.count;
            getEl('detect-time').textContent = `${result.inference_ms}ms`;

            // Class breakdown
            renderClassBreakdown(result.class_summary);

        } catch (e) { /* error shown */ } finally {
            btn.disabled = false;
            btn.textContent = 'RUN DETECTION';
        }
    }

    function renderClassBreakdown(summary) {
        const container = getEl('class-breakdown');
        container.innerHTML = '';

        const maxCount = Math.max(...Object.values(summary), 1);

        Object.entries(summary).forEach(([name, count], i) => {
            const color = CLASS_COLORS[i % CLASS_COLORS.length];
            const pct = (count / maxCount * 100).toFixed(0);

            const row = document.createElement('div');
            row.className = 'class-bar-row';
            row.innerHTML = `
                <span class="class-bar-name">${name}</span>
                <div class="class-bar"><div class="class-bar-fill" style="width:${pct}%;background:${color}"></div></div>
                <span class="class-bar-count">${count}</span>
            `;
            container.appendChild(row);
        });
    }

    function updateExportButton() {
        const btn = getEl('btn-export-model');
        if (!btn) return;
        const hasModel = pipelineState.trainedModelPath || pipelineState.trainingComplete;
        btn.disabled = false; // Always allow — will fall back to pretrained
    }

    function exportModel() {
        // Direct download via browser
        const link = document.createElement('a');
        link.href = 'api/model/export';
        link.download = 'hermes-model.pt';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        showToast('Downloading model...', 'success');
    }

    return { init };
})();
