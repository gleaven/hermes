/* ============================================================
   HERMES — Main Application Controller
   ============================================================ */

// --- EventBus ---
const EventBus = (() => {
    const listeners = {};
    return {
        on(event, fn) {
            (listeners[event] = listeners[event] || []).push(fn);
        },
        off(event, fn) {
            listeners[event] = (listeners[event] || []).filter(f => f !== fn);
        },
        emit(event, data) {
            (listeners[event] || []).forEach(fn => fn(data));
        }
    };
})();

// --- Pipeline State ---
const pipelineState = {
    currentStage: 0,
    maxUnlockedStage: 1,
    selectedDataset: null,
    classes: [],
    labels: {},
    augConfig: { augmentations: ['flip_h', 'brightness', 'blur'], multiplier: 3 },
    trainingConfig: { model: 'yolo11n', epochs: 50, imgsz: 640, batch: 16 },
    trainingJobId: null,
    trainedModelPath: null,
    trainingComplete: false,
    selectedDetectImage: null,
};

// --- Class Colors ---
const CLASS_COLORS = [
    '#00ff88', '#00f0ff', '#ffaa00', '#ff3366',
    '#7b2dff', '#ff00aa', '#00b4d8', '#76b900',
    '#ff6600', '#00ffcc', '#ff44cc', '#88ff00',
];

// --- API Helper ---
async function api(url, options = {}) {
    try {
        const resp = await fetch(url, options);
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ error: resp.statusText }));
            throw new Error(err.error || resp.statusText);
        }
        return await resp.json();
    } catch (e) {
        showToast(e.message, 'error');
        throw e;
    }
}

// --- Toast Notifications ---
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 4000);
}

// --- Stage Navigation ---
function goToStage(n) {
    if (n < 0 || n > 6) return;
    if (n > pipelineState.maxUnlockedStage) return;

    const prev = pipelineState.currentStage;
    pipelineState.currentStage = n;

    // Update stage panels
    document.querySelectorAll('.stage-panel').forEach(p => p.classList.remove('active'));
    const panel = document.getElementById(`stage-${n}`);
    if (panel) panel.classList.add('active');

    // Update sidebar
    document.querySelectorAll('.phase-btn').forEach(btn => {
        const s = parseInt(btn.dataset.stage);
        btn.classList.remove('active', 'completed', 'locked');
        if (s === n) btn.classList.add('active');
        else if (s < pipelineState.maxUnlockedStage) btn.classList.add('completed');
        else if (s > pipelineState.maxUnlockedStage) btn.classList.add('locked');
    });

    // Update header pipeline nodes
    document.querySelectorAll('.pip-node').forEach(node => {
        const s = parseInt(node.dataset.stage);
        node.classList.remove('active', 'completed');
        if (s === n) node.classList.add('active');
        else if (s < n) node.classList.add('completed');
    });
    document.querySelectorAll('.pip-line').forEach((line, i) => {
        line.classList.toggle('active', i < n);
    });

    // Update metrics panel
    document.getElementById('metric-phase').textContent = `${n} / 6`;
    document.querySelectorAll('.mini-stage').forEach((ms, i) => {
        ms.classList.remove('active', 'completed');
        if (i === n) ms.classList.add('active');
        else if (i < n) ms.classList.add('completed');
    });

    EventBus.emit('stage:entered', n);
}

function unlockStage(n) {
    if (n > pipelineState.maxUnlockedStage) {
        pipelineState.maxUnlockedStage = n;
        // Update sidebar buttons
        document.querySelectorAll('.phase-btn').forEach(btn => {
            const s = parseInt(btn.dataset.stage);
            if (s <= n && !btn.classList.contains('active')) {
                btn.classList.remove('locked');
            }
        });
    }
}

// --- Metrics Panel Updates ---
function updateMetrics() {
    const ds = pipelineState.selectedDataset;
    if (ds) {
        document.getElementById('metric-dataset').textContent = ds.name || '--';
        document.getElementById('metric-images').textContent = ds.image_count || '--';
        document.getElementById('metric-classes').textContent = (pipelineState.classes || []).length || '--';
    }
    document.getElementById('metric-labels').textContent = Object.keys(pipelineState.labels).length || '--';
    document.getElementById('metric-model').textContent = pipelineState.trainingConfig.model.toUpperCase();
}

// --- GPU Polling ---
let gpuPollInterval = null;
function startGpuPolling() {
    const update = async () => {
        try {
            const data = await fetch('api/gpu').then(r => r.json());
            if (data.available) {
                const pct = data.utilization_pct || 0;
                document.getElementById('gpu-fill').style.width = `${pct}%`;
                document.getElementById('gpu-pct').textContent = `${pct}%`;
                if (data.name) {
                    document.getElementById('gpu-label').textContent = data.name.replace('NVIDIA ', '');
                }
            }
        } catch (e) { /* ignore */ }
    };
    update();
    gpuPollInterval = setInterval(update, 5000);
}

// --- Initialization ---
function initApp() {
    // Wire sidebar buttons
    document.querySelectorAll('.phase-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const s = parseInt(btn.dataset.stage);
            if (s <= pipelineState.maxUnlockedStage) goToStage(s);
        });
    });

    // Wire header pipeline nodes
    document.querySelectorAll('.pip-node').forEach(node => {
        node.addEventListener('click', () => {
            const s = parseInt(node.dataset.stage);
            if (s <= pipelineState.maxUnlockedStage) goToStage(s);
        });
    });

    // Wire navigation buttons
    document.getElementById('btn-next-0')?.addEventListener('click', () => { unlockStage(1); goToStage(1); });
    document.getElementById('btn-prev-1')?.addEventListener('click', () => goToStage(0));
    document.getElementById('btn-next-1')?.addEventListener('click', () => goToStage(2));
    document.getElementById('btn-prev-2')?.addEventListener('click', () => goToStage(1));
    document.getElementById('btn-next-2')?.addEventListener('click', () => goToStage(3));
    document.getElementById('btn-prev-3')?.addEventListener('click', () => goToStage(2));
    document.getElementById('btn-next-3')?.addEventListener('click', () => goToStage(4));
    document.getElementById('btn-prev-4')?.addEventListener('click', () => goToStage(3));
    document.getElementById('btn-next-4')?.addEventListener('click', () => { unlockStage(5); goToStage(5); });
    document.getElementById('btn-prev-5')?.addEventListener('click', () => goToStage(4));
    document.getElementById('btn-next-5')?.addEventListener('click', () => { unlockStage(6); goToStage(6); });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        const num = parseInt(e.key);
        if (num >= 0 && num <= 6) goToStage(num);
    });

    // Initialize all stage modules
    SatelliteStage.init();
    CollectionStage.init();
    LabelingStage.init();
    AugmentationStage.init();
    QualityStage.init();
    TrainingStage.init();
    AssessmentStage.init();
    DemoMode.init();

    // Start GPU polling
    startGpuPolling();

    // Hide loading overlay
    setTimeout(() => {
        document.getElementById('loading-overlay').classList.add('hidden');
    }, 1500);

    // Set initial stage
    goToStage(0);
}

document.addEventListener('DOMContentLoaded', initApp);
