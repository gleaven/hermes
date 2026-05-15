/* ============================================================
   Stage 2: ANNOTATION — Canvas Bounding Box Labeling
   ============================================================ */
const LabelingStage = (() => {
    let canvas, ctx;
    let currentImage = null;
    let currentImageEl = null;
    let imageList = [];
    let currentIndex = -1;
    let activeClassId = 0;
    let boxes = []; // [{class_id, x, y, w, h}] in YOLO normalized format
    let allLabels = {}; // {imageStem: [{class_id, x, y, w, h}]}
    let drawing = false;
    let startX = 0, startY = 0;
    let imgScale = 1, imgOffsetX = 0, imgOffsetY = 0;
    let imgW = 0, imgH = 0;
    let filterClassId = null; // null = no filter, number = filter by this class
    let filteredIndices = []; // image indices matching the filter

    function getEl(id) { return document.getElementById(id); }

    function init() {
        canvas = getEl('label-canvas');
        ctx = canvas.getContext('2d');

        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);
        canvas.addEventListener('contextmenu', onRightClick);

        getEl('btn-add-class')?.addEventListener('click', addClass);
        getEl('btn-auto-label')?.addEventListener('click', autoLabel);
        getEl('btn-save-labels')?.addEventListener('click', saveLabels);

        // Image navigator prev/next
        getEl('nav-prev')?.addEventListener('click', () => navigateImage(-1));
        getEl('nav-next')?.addEventListener('click', () => navigateImage(1));

        // Keyboard navigation (left/right arrows)
        document.addEventListener('keydown', (e) => {
            if (pipelineState.currentStage !== 2) return;
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
            if (e.key === 'ArrowLeft') { e.preventDefault(); navigateImage(-1); }
            else if (e.key === 'ArrowRight') { e.preventDefault(); navigateImage(1); }
        });

        EventBus.on('stage:entered', stage => {
            if (stage === 2) onEnter();
        });
    }

    function navigateImage(delta) {
        const indices = filterClassId !== null ? filteredIndices : imageList.map((_, i) => i);
        if (indices.length === 0) return;
        const curPos = indices.indexOf(currentIndex);
        let nextPos = curPos + delta;
        if (nextPos < 0) nextPos = indices.length - 1;
        if (nextPos >= indices.length) nextPos = 0;
        loadImage(indices[nextPos]);
    }

    async function onEnter() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        // Load images
        const data = await api(`api/datasets/${ds.id}/images?per_page=500`);
        imageList = data.images;

        // Load existing labels
        try {
            allLabels = await api(`api/labels/${ds.id}`);
            pipelineState.labels = allLabels;
        } catch (e) {
            allLabels = {};
        }

        // Set up default classes if none exist
        if (pipelineState.classes.length === 0) {
            pipelineState.classes = [
                { id: 0, name: 'object', color: CLASS_COLORS[0] }
            ];
        }

        renderClassPalette();
        renderNavigator();
        updateStats();
        checkReadyToAdvance();

        // Load first image
        if (imageList.length > 0) {
            loadImage(0);
        }
    }

    function renderClassPalette() {
        const palette = getEl('class-palette');
        palette.innerHTML = '';
        pipelineState.classes.forEach((cls, i) => {
            const color = cls.color || CLASS_COLORS[i % CLASS_COLORS.length];
            cls.color = color;
            const badge = document.createElement('div');
            const isFiltered = filterClassId === cls.id;
            badge.className = `class-badge ${cls.id === activeClassId ? 'active' : ''} ${isFiltered ? 'filtering' : ''}`;
            badge.style.color = color;
            badge.innerHTML = `
                <div class="class-dot" style="background:${color}"></div>
                <span class="class-name">${cls.name}</span>
                <span class="class-count" id="class-count-${cls.id}" title="Click to filter images with this class">0</span>
                <button class="class-filter-btn" title="Filter images containing ${cls.name}">&#9906;</button>
                <button class="class-delete-btn" title="Delete all ${cls.name} annotations">&times;</button>
            `;
            // Click badge body = select as active drawing class
            badge.addEventListener('click', (e) => {
                if (e.target.closest('.class-delete-btn') || e.target.closest('.class-filter-btn')) return;
                activeClassId = cls.id;
                palette.querySelectorAll('.class-badge').forEach(b => b.classList.remove('active'));
                badge.classList.add('active');
            });
            // Filter button
            badge.querySelector('.class-filter-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                toggleFilter(cls.id);
            });
            badge.querySelector('.class-delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                bulkDeleteClass(cls.id, cls.name);
            });
            palette.appendChild(badge);
        });
    }

    function toggleFilter(classId) {
        if (filterClassId === classId) {
            // Clear filter
            clearFilter();
        } else {
            // Set filter
            filterClassId = classId;
            updateFilteredIndices();
            renderClassPalette();
            renderFilterBar();
            renderNavigator();
            // Jump to first filtered image
            if (filteredIndices.length > 0) {
                loadImage(filteredIndices[0]);
            }
        }
    }

    function clearFilter() {
        filterClassId = null;
        filteredIndices = [];
        renderClassPalette();
        renderFilterBar();
        renderNavigator();
    }

    function updateFilteredIndices() {
        filteredIndices = [];
        if (filterClassId === null) return;
        imageList.forEach((img, i) => {
            const stem = stemOf(img.name);
            const labels = allLabels[stem] || [];
            if (labels.some(b => b.class_id === filterClassId)) {
                filteredIndices.push(i);
            }
        });
    }

    function renderFilterBar() {
        let bar = getEl('filter-bar');
        if (filterClassId === null) {
            if (bar) bar.style.display = 'none';
            return;
        }

        const cls = pipelineState.classes.find(c => c.id === filterClassId);
        if (!cls) return;

        if (!bar) {
            bar = document.createElement('div');
            bar.id = 'filter-bar';
            bar.className = 'filter-bar';
            const nav = getEl('image-navigator');
            nav.parentNode.insertBefore(bar, nav);
        }

        bar.style.display = 'flex';
        bar.innerHTML = `
            <span class="filter-indicator">
                <span class="filter-icon">&#9906;</span>
                Showing <strong>${filteredIndices.length}</strong> image${filteredIndices.length !== 1 ? 's' : ''} with
                <span class="filter-class-tag" style="color:${cls.color};border-color:${cls.color}">
                    <span class="class-dot" style="background:${cls.color}"></span>
                    ${cls.name}
                </span>
            </span>
            <button class="btn-clear-filter" title="Clear filter">&times; CLEAR FILTER</button>
        `;
        bar.querySelector('.btn-clear-filter').addEventListener('click', clearFilter);
    }

    function bulkDeleteClass(classId, className) {
        // Count total annotations of this class
        let count = 0;
        Object.values(allLabels).forEach(bxs => {
            bxs.forEach(b => { if (b.class_id === classId) count++; });
        });

        if (count === 0) {
            showToast(`No ${className} annotations to remove`, 'info');
            return;
        }

        if (!confirm(`Remove all ${count} "${className}" annotations across the entire dataset?`)) return;

        // Remove all annotations with this class_id
        Object.keys(allLabels).forEach(stem => {
            allLabels[stem] = allLabels[stem].filter(b => b.class_id !== classId);
        });
        // Clean up empty entries
        Object.keys(allLabels).forEach(stem => {
            if (allLabels[stem].length === 0) delete allLabels[stem];
        });
        pipelineState.labels = allLabels;

        // Remove the class from the class list
        pipelineState.classes = pipelineState.classes.filter(c => c.id !== classId);
        if (activeClassId === classId) {
            activeClassId = pipelineState.classes.length > 0 ? pipelineState.classes[0].id : 0;
        }

        // Clear filter if it was on this class
        if (filterClassId === classId) {
            filterClassId = null;
            filteredIndices = [];
        }

        // Refresh current image boxes
        if (currentImage) {
            const stem = stemOf(currentImage.name);
            boxes = allLabels[stem] || [];
            draw();
            renderLabelList();
        }

        renderClassPalette();
        updateStats();
        renderFilterBar();
        renderNavigator();
        checkReadyToAdvance();
        saveLabels();
        showToast(`Removed ${count} "${className}" annotations`, 'success');
    }

    function addClass() {
        const name = prompt('Class name:');
        if (!name) return;
        const id = pipelineState.classes.length;
        pipelineState.classes.push({
            id, name: name.trim(),
            color: CLASS_COLORS[id % CLASS_COLORS.length]
        });
        renderClassPalette();
        updateMetrics();
    }

    function renderNavigator() {
        const nav = getEl('image-navigator');
        nav.innerHTML = '';
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        const indicesToShow = filterClassId !== null ? filteredIndices : imageList.map((_, i) => i);

        indicesToShow.forEach(i => {
            const img = imageList[i];
            const thumb = document.createElement('div');
            thumb.className = `nav-thumb ${i === currentIndex ? 'active' : ''} ${img.labeled || allLabels[stemOf(img.name)] ? 'labeled' : ''}`;
            thumb.innerHTML = `<img src="api/datasets/${ds.id}/images/${img.name}" loading="lazy" alt="">`;
            thumb.addEventListener('click', () => loadImage(i));
            nav.appendChild(thumb);
        });
    }

    function stemOf(name) {
        return name.replace(/\.[^.]+$/, '');
    }

    async function loadImage(index) {
        const ds = pipelineState.selectedDataset;
        if (!ds || !imageList[index]) return;

        currentIndex = index;
        currentImage = imageList[index];
        const stem = stemOf(currentImage.name);

        // Load boxes for this image
        boxes = allLabels[stem] || [];

        // Load image
        getEl('canvas-instructions').style.display = 'none';
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            currentImageEl = img;
            imgW = img.naturalWidth;
            imgH = img.naturalHeight;
            resizeCanvas();
            draw();
        };
        img.src = `api/datasets/${ds.id}/images/${currentImage.name}`;

        // Update navigator active state
        document.querySelectorAll('#image-navigator .nav-thumb').forEach(t => t.classList.remove('active'));
        const indicesToShow = filterClassId !== null ? filteredIndices : imageList.map((_, i) => i);
        const thumbPos = indicesToShow.indexOf(index);
        const thumbs = document.querySelectorAll('#image-navigator .nav-thumb');
        if (thumbPos >= 0 && thumbs[thumbPos]) {
            thumbs[thumbPos].classList.add('active');
            // Auto-scroll thumb into view
            thumbs[thumbPos].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }

        // Update counter
        const counter = getEl('nav-counter');
        if (counter) {
            counter.textContent = `${thumbPos + 1} / ${indicesToShow.length}`;
        }

        renderLabelList();
    }

    function resizeCanvas() {
        const container = getEl('canvas-container');
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        canvas.width = cw;
        canvas.height = ch;

        // Fit image in canvas
        const scaleX = cw / imgW;
        const scaleY = ch / imgH;
        imgScale = Math.min(scaleX, scaleY);
        imgOffsetX = (cw - imgW * imgScale) / 2;
        imgOffsetY = (ch - imgH * imgScale) / 2;
    }

    function draw() {
        if (!currentImageEl) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw image
        ctx.drawImage(currentImageEl, imgOffsetX, imgOffsetY, imgW * imgScale, imgH * imgScale);

        // Draw boxes
        boxes.forEach((box, i) => {
            const cls = pipelineState.classes.find(c => c.id === box.class_id);
            const color = cls?.color || CLASS_COLORS[box.class_id % CLASS_COLORS.length];
            const [px, py, pw, ph] = yoloToPixel(box);

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(px, py, pw, ph);

            // Label background
            const label = cls?.name || `class_${box.class_id}`;
            ctx.font = '11px "Share Tech Mono", monospace';
            const tw = ctx.measureText(label).width + 8;
            ctx.fillStyle = color;
            ctx.fillRect(px, py - 16, tw, 16);
            ctx.fillStyle = '#060610';
            ctx.fillText(label, px + 4, py - 4);
        });
    }

    function yoloToPixel(box) {
        const cx = box.x * imgW * imgScale + imgOffsetX;
        const cy = box.y * imgH * imgScale + imgOffsetY;
        const bw = box.w * imgW * imgScale;
        const bh = box.h * imgH * imgScale;
        return [cx - bw / 2, cy - bh / 2, bw, bh];
    }

    function pixelToYolo(x1, y1, x2, y2) {
        const nx1 = (Math.min(x1, x2) - imgOffsetX) / (imgW * imgScale);
        const ny1 = (Math.min(y1, y2) - imgOffsetY) / (imgH * imgScale);
        const nx2 = (Math.max(x1, x2) - imgOffsetX) / (imgW * imgScale);
        const ny2 = (Math.max(y1, y2) - imgOffsetY) / (imgH * imgScale);

        const cx = (nx1 + nx2) / 2;
        const cy = (ny1 + ny2) / 2;
        const w = nx2 - nx1;
        const h = ny2 - ny1;

        // Clamp to [0, 1]
        return {
            class_id: activeClassId,
            x: Math.max(0, Math.min(1, cx)),
            y: Math.max(0, Math.min(1, cy)),
            w: Math.max(0.001, Math.min(1, w)),
            h: Math.max(0.001, Math.min(1, h)),
        };
    }

    function onMouseDown(e) {
        if (e.button !== 0) return;
        drawing = true;
        const rect = canvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
    }

    function onMouseMove(e) {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        draw();
        // Draw preview rectangle
        ctx.strokeStyle = pipelineState.classes.find(c => c.id === activeClassId)?.color || CLASS_COLORS[0];
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startX, startY, x - startX, y - startY);
        ctx.setLineDash([]);
    }

    function onMouseUp(e) {
        if (!drawing) return;
        drawing = false;
        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;

        // Minimum box size
        if (Math.abs(endX - startX) < 5 || Math.abs(endY - startY) < 5) {
            draw();
            return;
        }

        const box = pixelToYolo(startX, startY, endX, endY);
        boxes.push(box);

        // Save to allLabels
        const stem = stemOf(currentImage.name);
        allLabels[stem] = [...boxes];
        pipelineState.labels = allLabels;

        draw();
        renderLabelList();
        updateStats();
        checkReadyToAdvance();
    }

    function onRightClick(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        // Find and remove box under cursor
        for (let i = boxes.length - 1; i >= 0; i--) {
            const [px, py, pw, ph] = yoloToPixel(boxes[i]);
            if (mx >= px && mx <= px + pw && my >= py && my <= py + ph) {
                boxes.splice(i, 1);
                const stem = stemOf(currentImage.name);
                allLabels[stem] = [...boxes];
                pipelineState.labels = allLabels;
                draw();
                renderLabelList();
                updateStats();
                break;
            }
        }
    }

    function relabelBox(boxIndex, newClassId) {
        if (boxIndex < 0 || boxIndex >= boxes.length) return;
        boxes[boxIndex].class_id = newClassId;
        const stem = stemOf(currentImage.name);
        allLabels[stem] = [...boxes];
        pipelineState.labels = allLabels;
        draw();
        renderLabelList();
        updateStats();
    }

    function renderLabelList() {
        const list = getEl('label-list');
        list.innerHTML = '';
        boxes.forEach((box, i) => {
            const cls = pipelineState.classes.find(c => c.id === box.class_id);
            const color = cls?.color || CLASS_COLORS[box.class_id % CLASS_COLORS.length];
            const item = document.createElement('div');
            item.className = 'label-item';
            item.style.borderLeft = `3px solid ${color}`;

            // Build class selector dropdown
            const options = pipelineState.classes.map(c => {
                const sel = c.id === box.class_id ? 'selected' : '';
                return `<option value="${c.id}" ${sel} style="color:${c.color}">${c.name}</option>`;
            }).join('');

            item.innerHTML = `
                <select class="label-class-select" title="Change class">
                    ${options}
                </select>
                <span class="del-btn" title="Delete">&times;</span>
            `;
            item.querySelector('.label-class-select').addEventListener('change', (e) => {
                relabelBox(i, parseInt(e.target.value));
            });
            item.querySelector('.del-btn').addEventListener('click', () => {
                boxes.splice(i, 1);
                const stem = stemOf(currentImage.name);
                allLabels[stem] = [...boxes];
                draw();
                renderLabelList();
                updateStats();
            });
            list.appendChild(item);
        });
    }

    function updateStats() {
        const totalImages = imageList.length;
        const labeledImages = Object.keys(allLabels).filter(k => allLabels[k].length > 0).length;
        const totalBoxes = Object.values(allLabels).reduce((s, b) => s + b.length, 0);

        getEl('stat-labeled').textContent = `${labeledImages} / ${totalImages}`;
        getEl('stat-boxes').textContent = totalBoxes;
        document.getElementById('metric-labels').textContent = labeledImages;

        // Update per-class counts in palette
        const classCounts = {};
        Object.values(allLabels).forEach(bxs => {
            bxs.forEach(b => { classCounts[b.class_id] = (classCounts[b.class_id] || 0) + 1; });
        });
        pipelineState.classes.forEach(cls => {
            const el = document.getElementById(`class-count-${cls.id}`);
            if (el) el.textContent = classCounts[cls.id] || 0;
        });

        // Update filter indices if filter is active
        if (filterClassId !== null) {
            updateFilteredIndices();
            renderFilterBar();
        }
    }

    function checkReadyToAdvance() {
        const labeledCount = Object.keys(allLabels).filter(k => allLabels[k].length > 0).length;
        getEl('btn-next-2').disabled = labeledCount < 1;
        if (labeledCount > 0) unlockStage(3);
    }

    async function autoLabel() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        showToast('Running auto-labeling with YOLO...', 'info');
        getEl('btn-auto-label').disabled = true;
        getEl('btn-auto-label').textContent = 'LABELING...';

        try {
            const result = await api(`api/labels/${ds.id}/auto`, { method: 'POST' });
            showToast(`Auto-labeled ${result.labeled} images with ${result.total_boxes} boxes`, 'success');

            // Update classes from model
            if (result.classes?.length) {
                pipelineState.classes = result.classes.map((c, i) => ({
                    ...c, color: CLASS_COLORS[i % CLASS_COLORS.length]
                }));
                renderClassPalette();
            }

            // Reload labels
            allLabels = await api(`api/labels/${ds.id}`);
            pipelineState.labels = allLabels;

            // Reload current image
            if (currentIndex >= 0) {
                const stem = stemOf(imageList[currentIndex].name);
                boxes = allLabels[stem] || [];
                draw();
                renderLabelList();
            }

            updateStats();
            renderNavigator();
            checkReadyToAdvance();
            updateMetrics();
        } catch (e) {
            // Error shown by api()
        } finally {
            getEl('btn-auto-label').disabled = false;
            getEl('btn-auto-label').textContent = 'AUTO-LABEL';
        }
    }

    async function saveLabels() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        try {
            await api(`api/labels/${ds.id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    labels: allLabels,
                    classes: pipelineState.classes,
                }),
            });
            showToast('Labels saved', 'success');
        } catch (e) { /* error shown */ }
    }

    // Handle canvas resize
    window.addEventListener('resize', () => {
        if (currentImageEl) {
            resizeCanvas();
            draw();
        }
    });

    return { init };
})();
