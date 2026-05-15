/* ============================================================
   Stage 5: MODEL TRAINING — YOLO11 Live Training Dashboard
   ============================================================ */
const TrainingStage = (() => {
    let ws = null;
    let lossChart = null;
    let mapChart = null;
    let isTraining = false;

    function getEl(id) { return document.getElementById(id); }

    function init() {
        setupConfig();

        getEl('btn-start-training')?.addEventListener('click', startTraining);
        getEl('btn-retrain')?.addEventListener('click', showConfig);
        getEl('btn-clear-training')?.addEventListener('click', clearHistory);

        EventBus.on('stage:entered', stage => {
            if (stage === 5) onEnter();
        });
    }

    async function onEnter() {
        if (isTraining) return; // Don't interrupt active training

        // Check for saved training results
        try {
            const hist = await api('api/train/history');
            if (hist.runs && hist.runs.length > 0) {
                const latest = hist.runs[0];
                // Restore state from saved results
                pipelineState.trainingComplete = true;
                pipelineState.trainedModelPath = latest.model_path;

                showDashboard();
                restoreResults(latest);
                return;
            }
        } catch (e) { /* no history */ }

        // No saved results — show config
        showConfig();
    }

    function setupConfig() {
        // Model selector
        document.querySelectorAll('.model-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                pipelineState.trainingConfig.model = btn.dataset.model;
            });
        });

        // Epochs slider
        const slider = getEl('epochs-slider');
        const value = getEl('epochs-value');
        slider?.addEventListener('input', () => {
            value.textContent = slider.value;
            pipelineState.trainingConfig.epochs = parseInt(slider.value);
        });

        // Image size
        document.querySelectorAll('.size-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.size-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                pipelineState.trainingConfig.imgsz = parseInt(btn.dataset.size);
            });
        });

        // Batch size
        document.querySelectorAll('.batch-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.batch-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                pipelineState.trainingConfig.batch = parseInt(btn.dataset.batch);
            });
        });
    }

    function showConfig() {
        isTraining = false;
        getEl('training-config').style.display = 'flex';
        getEl('training-dashboard').style.display = 'none';
        getEl('training-actions')?.style.setProperty('display', 'none');
    }

    async function startTraining() {
        const ds = pipelineState.selectedDataset;
        if (!ds) return;

        // Save labels first
        try {
            await api(`api/labels/${ds.id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    labels: pipelineState.labels,
                    classes: pipelineState.classes,
                }),
            });
        } catch (e) {
            showToast('Failed to save labels before training', 'error');
            return;
        }

        const config = pipelineState.trainingConfig;
        isTraining = true;
        pipelineState.trainingComplete = false;

        try {
            const result = await api('api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: ds.id,
                    model: config.model,
                    epochs: config.epochs,
                    imgsz: config.imgsz,
                    batch: config.batch,
                }),
            });

            pipelineState.trainingJobId = result.job_id;
            showToast('Training started', 'success');
            showDashboard();
            hideTrainingActions();
            connectWebSocket(result.job_id);

        } catch (e) { /* error shown */ isTraining = false; }
    }

    function showDashboard() {
        getEl('training-config').style.display = 'none';
        getEl('training-dashboard').style.display = 'flex';
        initCharts();
    }

    function showTrainingActions() {
        const actions = getEl('training-actions');
        if (actions) actions.style.display = 'flex';
    }

    function hideTrainingActions() {
        const actions = getEl('training-actions');
        if (actions) actions.style.display = 'none';
    }

    function restoreResults(run) {
        // Populate charts with saved history
        const h = run.history || {};
        const epochs = h.box_loss?.map((_, i) => i + 1) || [];

        if (lossChart) {
            lossChart.data.labels = epochs;
            lossChart.data.datasets[0].data = h.box_loss || [];
            lossChart.data.datasets[1].data = h.cls_loss || [];
            lossChart.data.datasets[2].data = h.dfl_loss || [];
            lossChart.update('none');
        }

        if (mapChart) {
            mapChart.data.labels = epochs;
            mapChart.data.datasets[0].data = h.mAP50 || [];
            mapChart.data.datasets[1].data = h.mAP50_95 || [];
            mapChart.update('none');
        }

        // Populate stats
        const f = run.final || {};
        const totalEpochs = run.config?.epochs || epochs.length;
        getEl('training-progress-fill').style.width = '100%';
        getEl('training-progress-text').textContent = '100%';
        getEl('train-epoch').textContent = `${totalEpochs} / ${totalEpochs}`;
        getEl('train-map50').textContent = (f.mAP50 || 0).toFixed(3);
        getEl('train-map95').textContent = (f.mAP50_95 || 0).toFixed(3);

        const elapsed = run.elapsed_seconds || 0;
        const min = Math.floor(elapsed / 60);
        const sec = elapsed % 60;
        getEl('train-eta').textContent = `${min}:${String(sec).padStart(2, '0')}`;

        // Update metrics panel
        document.getElementById('metric-map').textContent = (f.mAP50 || 0).toFixed(3);

        // Enable proceed button
        getEl('btn-next-5').disabled = false;
        unlockStage(6);

        // Show retrain / clear buttons
        showTrainingActions();
    }

    function initCharts() {
        const chartOpts = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: 'rgba(255,255,255,0.6)', font: { family: "'Share Tech Mono', monospace", size: 10 } } } },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: 'rgba(255,255,255,0.4)', font: { family: "'Share Tech Mono', monospace", size: 9 } },
                    title: { display: true, text: 'Epoch', color: 'rgba(255,255,255,0.4)', font: { size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: 'rgba(255,255,255,0.4)', font: { family: "'Share Tech Mono', monospace", size: 9 } }
                }
            }
        };

        // Loss chart
        const lossCanvas = getEl('chart-loss');
        if (lossChart) lossChart.destroy();
        lossChart = new Chart(lossCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Box Loss', data: [], borderColor: '#00ff88', borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.3 },
                    { label: 'Cls Loss', data: [], borderColor: '#00f0ff', borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.3 },
                    { label: 'DFL Loss', data: [], borderColor: '#ffaa00', borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.3 },
                ]
            },
            options: chartOpts,
        });

        // mAP chart
        const mapCanvas = getEl('chart-map');
        if (mapChart) mapChart.destroy();
        mapChart = new Chart(mapCanvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'mAP@50', data: [], borderColor: '#00ff88', borderWidth: 2, fill: true, backgroundColor: 'rgba(0,255,136,0.1)', pointRadius: 0, tension: 0.3 },
                    { label: 'mAP@50-95', data: [], borderColor: '#7b2dff', borderWidth: 1.5, fill: false, pointRadius: 0, tension: 0.3 },
                ]
            },
            options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, min: 0, max: 1 } } },
        });
    }

    function connectWebSocket(jobId) {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/hermes/ws/train/${jobId}`);

        ws.onmessage = (event) => {
            const state = JSON.parse(event.data);
            updateDashboard(state);
        };

        ws.onclose = () => {
            // Fallback to polling
            if (isTraining && !pipelineState.trainingComplete) {
                setTimeout(() => pollTraining(jobId), 2000);
            }
        };

        ws.onerror = () => {
            ws.close();
        };
    }

    async function pollTraining(jobId) {
        if (pipelineState.trainingComplete || !isTraining) return;
        try {
            const state = await api(`api/train/${jobId}`);
            updateDashboard(state);
            if (state.status !== 'completed' && state.status !== 'failed') {
                setTimeout(() => pollTraining(jobId), 3000);
            }
        } catch (e) { /* retry */ setTimeout(() => pollTraining(jobId), 5000); }
    }

    function updateDashboard(state) {
        // Progress bar
        const progress = state.progress || 0;
        getEl('training-progress-fill').style.width = `${progress}%`;
        getEl('training-progress-text').textContent = `${progress}%`;

        // Stats
        getEl('train-epoch').textContent = `${state.epoch || 0} / ${state.total_epochs || 0}`;
        getEl('train-map50').textContent = (state.metrics?.mAP50 || 0).toFixed(3);
        getEl('train-map95').textContent = (state.metrics?.mAP50_95 || 0).toFixed(3);

        // ETA
        if (state.eta_seconds && state.eta_seconds > 0) {
            const min = Math.floor(state.eta_seconds / 60);
            const sec = state.eta_seconds % 60;
            getEl('train-eta').textContent = `${min}:${String(sec).padStart(2, '0')}`;
        }

        // Update charts
        if (state.history) {
            const epochs = state.history.box_loss?.map((_, i) => i + 1) || [];

            lossChart.data.labels = epochs;
            lossChart.data.datasets[0].data = state.history.box_loss || [];
            lossChart.data.datasets[1].data = state.history.cls_loss || [];
            lossChart.data.datasets[2].data = state.history.dfl_loss || [];
            lossChart.update('none');

            mapChart.data.labels = epochs;
            mapChart.data.datasets[0].data = state.history.mAP50 || [];
            mapChart.data.datasets[1].data = state.history.mAP50_95 || [];
            mapChart.update('none');
        }

        // Update metrics panel
        document.getElementById('metric-map').textContent = (state.metrics?.mAP50 || 0).toFixed(3);

        // Completion
        if (state.status === 'completed') {
            isTraining = false;
            pipelineState.trainingComplete = true;
            pipelineState.trainedModelPath = state.model_path;
            showToast('Training complete!', 'success');
            getEl('btn-next-5').disabled = false;
            unlockStage(6);

            // Show elapsed time instead of ETA
            if (state.elapsed_seconds) {
                const min = Math.floor(state.elapsed_seconds / 60);
                const sec = state.elapsed_seconds % 60;
                getEl('train-eta').textContent = `${min}:${String(sec).padStart(2, '0')}`;
            }

            // Update status dot
            document.getElementById('status-dot').style.background = '#00ff88';
            document.getElementById('status-dot').style.animation = 'none';

            // Show retrain / clear buttons
            showTrainingActions();

            if (ws) ws.close();
        } else if (state.status === 'failed') {
            isTraining = false;
            showToast(`Training failed: ${state.error}`, 'error');
            showConfig();
            if (ws) ws.close();
        }
    }

    async function clearHistory() {
        if (!confirm('Clear all training results? This cannot be undone.')) return;
        try {
            await api('api/train/history', { method: 'DELETE' });
            pipelineState.trainingComplete = false;
            pipelineState.trainedModelPath = null;
            pipelineState.trainingJobId = null;
            showToast('Training history cleared', 'success');
            showConfig();
        } catch (e) { /* error shown */ }
    }

    return { init, updateDashboard, initCharts, showDashboard };
})();
