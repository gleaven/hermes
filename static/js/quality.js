/* ============================================================
   Stage 4: QUALITY ASSURANCE — Dataset Statistics
   ============================================================ */
const QualityStage = (() => {
    let distChart = null;
    let cachedStats = null;
    let cachedImages = null;
    let allLabels = {};
    let activeClassFilter = null; // null = show all

    function getEl(id) { return document.getElementById(id); }

    function init() {
        EventBus.on('stage:entered', stage => {
            if (stage === 4) onEnter();
        });

        // Shuffle button
        getEl('btn-shuffle-samples')?.addEventListener('click', () => {
            if (cachedStats) shuffleSamples();
        });
    }

    async function onEnter() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        activeClassFilter = null;

        try {
            cachedStats = await api(`api/datasets/${ds.id}/stats`);

            // Load all images for shuffling
            const imgData = await api(`api/datasets/${ds.id}/images?per_page=500`);
            cachedImages = imgData.images;

            // Load labels from server if not already in client state
            if (!pipelineState.labels || Object.keys(pipelineState.labels).length === 0) {
                try {
                    allLabels = await api(`api/labels/${ds.id}`);
                    pipelineState.labels = allLabels;
                } catch (e) {
                    allLabels = {};
                }
            } else {
                allLabels = pipelineState.labels;
            }

            animateStats(cachedStats);
            renderDistribution(cachedStats);
            renderSamples(ds);
        } catch (e) { /* error shown */ }
    }

    // --- Animated Count-Up ---
    function animateStats(stats) {
        animateValue('qa-images', stats.image_count);
        animateValue('qa-labels', stats.total_boxes);
        animateValue('qa-avg', stats.avg_boxes_per_image, 1);
        animateValue('qa-balance', stats.balance_score, 0, '%');
    }

    function animateValue(elId, target, decimals = 0, suffix = '') {
        const el = getEl(elId);
        if (!el) return;
        const duration = 1200;
        const start = performance.now();
        const from = 0;

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = from + (target - from) * ease;
            el.textContent = decimals > 0
                ? current.toFixed(decimals) + suffix
                : Math.round(current) + suffix;
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // --- Class Distribution Chart ---
    function renderDistribution(stats) {
        const canvas = getEl('chart-distribution');
        if (!canvas) return;

        const classes = stats.classes || [];
        const dist = stats.class_distribution || {};

        const labels = [];
        const data = [];
        const colors = [];
        const classIds = [];

        Object.keys(dist).sort((a, b) => a - b).forEach((clsId, i) => {
            const cls = classes.find(c => c.id === parseInt(clsId));
            labels.push(cls?.name || `class_${clsId}`);
            data.push(dist[clsId]);
            colors.push(CLASS_COLORS[i % CLASS_COLORS.length]);
            classIds.push(parseInt(clsId));
        });

        if (distChart) distChart.destroy();

        distChart = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data,
                    backgroundColor: colors.map(c => c + '40'),
                    borderColor: colors,
                    borderWidth: 1,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.raw} annotations`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: {
                            color: 'rgba(255,255,255,0.5)',
                            font: { family: "'Share Tech Mono', monospace", size: 10 }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)',
                            font: { family: "'Rajdhani', sans-serif", size: 12 }
                        }
                    }
                },
                onClick: (evt, elements) => {
                    if (elements.length > 0) {
                        const idx = elements[0].index;
                        const clickedClassId = classIds[idx];
                        // Toggle filter
                        if (activeClassFilter === clickedClassId) {
                            activeClassFilter = null;
                            resetChartHighlight(colors);
                        } else {
                            activeClassFilter = clickedClassId;
                            highlightChartBar(idx, colors);
                        }
                        renderSamples(pipelineState.selectedDataset);
                        updateFilterBadge(labels[idx]);
                    }
                }
            }
        });
    }

    function highlightChartBar(activeIdx, colors) {
        if (!distChart) return;
        const ds = distChart.data.datasets[0];
        ds.backgroundColor = colors.map((c, i) => i === activeIdx ? c + '80' : c + '15');
        ds.borderColor = colors.map((c, i) => i === activeIdx ? c : c + '40');
        ds.borderWidth = colors.map((_, i) => i === activeIdx ? 2 : 1);
        distChart.update();
    }

    function resetChartHighlight(colors) {
        if (!distChart) return;
        const ds = distChart.data.datasets[0];
        ds.backgroundColor = colors.map(c => c + '40');
        ds.borderColor = colors;
        ds.borderWidth = 1;
        distChart.update();
    }

    function updateFilterBadge(className) {
        const badge = getEl('qa-filter-badge');
        if (!badge) return;
        if (activeClassFilter !== null) {
            badge.textContent = `Filtered: ${className}`;
            badge.style.display = 'inline-block';
        } else {
            badge.style.display = 'none';
        }
    }

    // --- Sample Grid ---
    function renderSamples(ds) {
        const grid = getEl('qa-sample-grid');
        grid.innerHTML = '';
        if (!cachedImages || !ds) return;

        // Filter images that have labels matching the active class
        let candidates = cachedImages;
        if (activeClassFilter !== null) {
            candidates = cachedImages.filter(img => {
                const stem = img.name.replace(/\.[^.]+$/, '');
                const boxes = allLabels[stem] || [];
                return boxes.some(b => b.class_id === activeClassFilter);
            });
        }

        // Pick 9 random samples
        const shuffled = [...candidates].sort(() => Math.random() - 0.5);
        const selected = shuffled.slice(0, 9);

        selected.forEach(img => {
            const sample = document.createElement('div');
            sample.className = 'qa-sample';

            const imgEl = document.createElement('img');
            imgEl.src = `api/datasets/${ds.id}/images/${encodeURIComponent(img.name)}`;
            imgEl.loading = 'lazy';
            sample.appendChild(imgEl);

            // Overlay canvas for boxes
            const stem = img.name.replace(/\.[^.]+$/, '');
            const boxes = allLabels[stem] || [];
            if (boxes.length > 0) {
                imgEl.onload = () => {
                    const cvs = document.createElement('canvas');
                    sample.appendChild(cvs);
                    const rect = sample.getBoundingClientRect();
                    cvs.width = rect.width;
                    cvs.height = rect.height;
                    const cctx = cvs.getContext('2d');

                    boxes.forEach(box => {
                        // Dim non-matching classes when filtered
                        if (activeClassFilter !== null && box.class_id !== activeClassFilter) {
                            cctx.globalAlpha = 0.15;
                        } else {
                            cctx.globalAlpha = 1;
                        }
                        const cls = pipelineState.classes.find(c => c.id === box.class_id);
                        const color = cls?.color || CLASS_COLORS[box.class_id % CLASS_COLORS.length];
                        const bx = (box.x - box.w / 2) * cvs.width;
                        const by = (box.y - box.h / 2) * cvs.height;
                        const bw = box.w * cvs.width;
                        const bh = box.h * cvs.height;

                        cctx.strokeStyle = color;
                        cctx.lineWidth = 1.5;
                        cctx.strokeRect(bx, by, bw, bh);
                    });
                    cctx.globalAlpha = 1;
                };
            }

            // Click to open lightbox
            sample.addEventListener('click', () => openLightbox(ds, img, boxes));

            grid.appendChild(sample);
        });
    }

    function shuffleSamples() {
        const ds = pipelineState.selectedDataset;
        if (ds) renderSamples(ds);
    }

    // --- Lightbox ---
    function openLightbox(ds, img, boxes) {
        // Remove existing lightbox
        document.getElementById('qa-lightbox')?.remove();

        const overlay = document.createElement('div');
        overlay.id = 'qa-lightbox';
        overlay.className = 'qa-lightbox';
        overlay.innerHTML = `
            <div class="qa-lightbox-backdrop"></div>
            <div class="qa-lightbox-content">
                <button class="qa-lightbox-close">&times;</button>
                <div class="qa-lightbox-image-wrap">
                    <img src="api/datasets/${ds.id}/images/${encodeURIComponent(img.name)}" alt="${img.name}">
                    <canvas id="qa-lightbox-canvas"></canvas>
                </div>
                <div class="qa-lightbox-info">
                    <div class="qa-lightbox-filename">${img.name}</div>
                    <div class="qa-lightbox-box-count">${boxes.length} annotation${boxes.length !== 1 ? 's' : ''}</div>
                    <div class="qa-lightbox-classes" id="qa-lightbox-classes"></div>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Draw boxes on lightbox canvas
        const lbImg = overlay.querySelector('img');
        const lbCanvas = overlay.querySelector('#qa-lightbox-canvas');
        lbImg.onload = () => {
            lbCanvas.width = lbImg.naturalWidth;
            lbCanvas.height = lbImg.naturalHeight;
            const ctx = lbCanvas.getContext('2d');

            const classCounts = {};
            boxes.forEach(box => {
                const cls = pipelineState.classes.find(c => c.id === box.class_id);
                const color = cls?.color || CLASS_COLORS[box.class_id % CLASS_COLORS.length];
                const name = cls?.name || `class_${box.class_id}`;
                classCounts[name] = (classCounts[name] || { count: 0, color });
                classCounts[name].count++;

                const bx = (box.x - box.w / 2) * lbCanvas.width;
                const by = (box.y - box.h / 2) * lbCanvas.height;
                const bw = box.w * lbCanvas.width;
                const bh = box.h * lbCanvas.height;

                ctx.strokeStyle = color;
                ctx.lineWidth = 3;
                ctx.strokeRect(bx, by, bw, bh);

                // Label
                ctx.font = 'bold 14px "Rajdhani", sans-serif';
                const tw = ctx.measureText(name).width + 10;
                ctx.fillStyle = color;
                ctx.fillRect(bx, by - 20, tw, 20);
                ctx.fillStyle = '#060610';
                ctx.fillText(name, bx + 5, by - 5);
            });

            // Render class breakdown
            const classEl = document.getElementById('qa-lightbox-classes');
            if (classEl) {
                classEl.innerHTML = Object.entries(classCounts).map(([name, { count, color }]) =>
                    `<span class="qa-lightbox-class-tag" style="border-color:${color};color:${color}">${name}: ${count}</span>`
                ).join('');
            }
        };

        // Close handlers
        overlay.querySelector('.qa-lightbox-close').addEventListener('click', () => overlay.remove());
        overlay.querySelector('.qa-lightbox-backdrop').addEventListener('click', () => overlay.remove());
        document.addEventListener('keydown', function escHandler(e) {
            if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', escHandler); }
        });
    }

    return { init };
})();
