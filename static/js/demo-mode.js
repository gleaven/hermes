/* ============================================================
   DEMO MODE — Auto-play Through All Stages
   ============================================================ */
const DemoMode = (() => {
    let active = false;
    let timer = null;

    function init() {
        const btn = document.getElementById('demo-mode-btn');
        btn?.addEventListener('click', toggle);
    }

    function toggle() {
        if (active) {
            stop();
        } else {
            start();
        }
    }

    async function start() {
        active = true;
        const btn = document.getElementById('demo-mode-btn');
        btn.classList.add('active');
        btn.querySelector('svg').innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';

        showToast('Demo mode started', 'info');

        // Check for pre-baked data
        let prebaked = null;
        try {
            prebaked = await api('api/demo/prebaked');
        } catch (e) { /* no prebaked data */ }

        // Auto-play through stages
        // Stage 0: show selector briefly, then satellite view
        unlockStage(0);
        goToStage(0);
        await delay(3000);
        if (!active) return;
        SatelliteStage.showView('satellite');
        await delay(4000);
        if (!active) return;

        await playStage(1, 5000);
        if (!active) return;

        await playStage(2, 6000);
        if (!active) return;

        await playStage(3, 6000);
        if (!active) return;

        await playStage(4, 5000);
        if (!active) return;

        // Stage 5: show pre-baked training curves if available
        if (prebaked?.available) {
            goToStage(5);
            unlockStage(5);
            TrainingStage.showDashboard();
            TrainingStage.initCharts();

            // Animate through pre-baked training history
            const history = prebaked.training_metrics?.history;
            if (history) {
                const totalEpochs = history.box_loss?.length || 0;
                for (let i = 1; i <= totalEpochs && active; i++) {
                    const state = {
                        status: i === totalEpochs ? 'completed' : 'training',
                        epoch: i,
                        total_epochs: totalEpochs,
                        progress: Math.round(i / totalEpochs * 100),
                        metrics: {
                            box_loss: history.box_loss[i - 1],
                            cls_loss: history.cls_loss[i - 1],
                            dfl_loss: history.dfl_loss[i - 1],
                            mAP50: history.mAP50[i - 1],
                            mAP50_95: history.mAP50_95[i - 1],
                        },
                        history: {
                            box_loss: history.box_loss.slice(0, i),
                            cls_loss: history.cls_loss.slice(0, i),
                            dfl_loss: history.dfl_loss.slice(0, i),
                            mAP50: history.mAP50.slice(0, i),
                            mAP50_95: history.mAP50_95.slice(0, i),
                        },
                        eta_seconds: 0,
                    };
                    TrainingStage.updateDashboard(state);
                    await delay(100); // Fast playback
                }
            }

            await delay(3000);
        } else {
            await playStage(5, 5000);
        }
        if (!active) return;

        await playStage(6, 8000);

        stop();
        showToast('Demo complete', 'success');
    }

    async function playStage(n, durationMs) {
        unlockStage(n);
        goToStage(n);
        await delay(durationMs);
    }

    function stop() {
        active = false;
        clearTimeout(timer);
        const btn = document.getElementById('demo-mode-btn');
        btn.classList.remove('active');
        btn.querySelector('svg').innerHTML = '<polygon points="5,3 19,12 5,21"/>';
    }

    function delay(ms) {
        return new Promise(resolve => {
            timer = setTimeout(resolve, ms);
        });
    }

    return { init };
})();
