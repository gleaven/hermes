/* ============================================================
   Stage 0: DATA SOURCE — Selector + Satellite Capture
   Multi-region capture with smart re-slicing and overlap.
   Supports drawing multiple regions and appending to datasets.
   ============================================================ */
const SatelliteStage = (() => {
    let map = null;
    let drawMode = false;
    let currentDrawingRect = null;
    let startLatLng = null;
    let currentZoom = 18;
    let outputTileSize = 640;
    let overlapPct = 0;
    let estimateData = null;
    let currentView = 'selector'; // 'selector' or 'satellite'
    let activeDatasetId = null;   // Set after first extraction, enables append

    // Multi-region state
    let regions = []; // Array of {id, rect, gridLayer, bounds, color}
    let nextRegionId = 0;
    const REGION_COLORS = ['#00ff88', '#ff3366', '#00f0ff', '#ffaa00', '#7b2dff', '#ff00aa'];

    const ESRI_TILE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
    const ZOOM_RESOLUTIONS = {15: 4.77, 16: 2.39, 17: 1.19, 18: 0.60, 19: 0.30};

    function getEl(id) { return document.getElementById(id); }

    function init() {
        // Source selector cards
        getEl('source-geo')?.addEventListener('click', () => showView('satellite'));
        getEl('source-datasets')?.addEventListener('click', () => {
            unlockStage(1);
            goToStage(1);
        });

        // Back to selector from satellite view
        getEl('btn-back-to-selector')?.addEventListener('click', () => showView('selector'));

        // Wire zoom selector buttons
        document.querySelectorAll('.sat-zoom-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.sat-zoom-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentZoom = parseInt(btn.dataset.zoom);
                getEl('sat-resolution').textContent = `${ZOOM_RESOLUTIONS[currentZoom]}m/px`;
                if (regions.length > 0) {
                    regions.forEach(r => showOutputGrid(r));
                    updateEstimate();
                }
            });
        });

        // Output tile size dropdown
        getEl('sat-tile-size')?.addEventListener('change', (e) => {
            outputTileSize = parseInt(e.target.value);
            if (regions.length > 0) {
                regions.forEach(r => showOutputGrid(r));
                updateEstimate();
            }
        });

        // Overlap slider
        getEl('sat-overlap')?.addEventListener('input', (e) => {
            overlapPct = parseInt(e.target.value);
            getEl('sat-overlap-value').textContent = `${overlapPct}%`;
            if (regions.length > 0) {
                regions.forEach(r => showOutputGrid(r));
                updateEstimate();
            }
        });

        getEl('btn-draw-region')?.addEventListener('click', toggleDrawMode);
        getEl('btn-clear-region')?.addEventListener('click', clearRegions);
        getEl('btn-extract-tiles')?.addEventListener('click', () => extractDataset(false));
        getEl('btn-append-tiles')?.addEventListener('click', () => extractDataset(true));

        EventBus.on('stage:entered', stage => {
            if (stage === 0) onEnter();
        });
    }

    function showView(view) {
        currentView = view;
        const selector = getEl('source-selector');
        const satellite = getEl('satellite-view');
        const actionsSelector = getEl('stage-0-actions-selector');
        const actionsSatellite = getEl('stage-0-actions-satellite');

        if (view === 'satellite') {
            selector.style.display = 'none';
            satellite.style.display = 'flex';
            actionsSelector.style.display = 'none';
            actionsSatellite.style.display = 'flex';
            if (!map) {
                setTimeout(initMap, 150);
            } else {
                setTimeout(() => map.invalidateSize(), 100);
            }
        } else {
            selector.style.display = 'flex';
            satellite.style.display = 'none';
            actionsSelector.style.display = '';
            actionsSatellite.style.display = 'none';
        }
    }

    function onEnter() {
        if (currentView === 'satellite' && map) {
            setTimeout(() => map.invalidateSize(), 100);
        }
        if (!pipelineState._satelliteDatasetId && currentView === 'selector') {
            showView('selector');
        }
    }

    function initMap() {
        map = L.map('satellite-map', {
            center: [37.7749, -122.4194],
            zoom: 16,
            minZoom: 3,
            maxZoom: 19,
            zoomControl: true,
            attributionControl: false,
        });

        L.tileLayer(ESRI_TILE_URL, {
            maxZoom: 19,
            attribution: 'Esri World Imagery',
        }).addTo(map);

        // Quick location buttons
        document.querySelectorAll('.sat-location-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const lat = parseFloat(btn.dataset.lat);
                const lng = parseFloat(btn.dataset.lng);
                const z = parseInt(btn.dataset.zoom || 16);
                map.flyTo([lat, lng], z, { duration: 1.5 });
            });
        });
    }

    // ======================
    // Drawing — stays in draw mode until user clicks DONE.
    // Map panning (scroll wheel, right-drag) works between draws.
    // Left-click-drag creates rectangles; dragging is only disabled
    // during the active stroke of a rectangle.
    // ======================

    function toggleDrawMode() {
        drawMode = !drawMode;
        const btn = getEl('btn-draw-region');
        const mapContainer = getEl('satellite-map');

        if (drawMode) {
            updateDrawButton();
            btn.classList.add('active');
            mapContainer.style.cursor = 'crosshair';
            // Don't disable dragging yet — only during active draw stroke
            map.on('mousedown', onDrawStart);
        } else {
            btn.textContent = 'DRAW CAPTURE REGION';
            btn.classList.remove('active');
            map.dragging.enable();
            mapContainer.style.cursor = '';
            map.off('mousedown', onDrawStart);
            map.off('mousemove', onDrawMove);
            map.off('mouseup', onDrawEnd);
            if (currentDrawingRect) {
                map.removeLayer(currentDrawingRect);
                currentDrawingRect = null;
            }
        }
    }

    function updateDrawButton() {
        const btn = getEl('btn-draw-region');
        if (drawMode) {
            const n = regions.length;
            btn.textContent = n > 0 ? `DONE (${n} REGION${n > 1 ? 'S' : ''})` : 'CANCEL DRAW';
        }
    }

    function onDrawStart(e) {
        if (e.originalEvent.button !== 0) return;
        startLatLng = e.latlng;

        // Disable dragging only during active rectangle stroke
        map.dragging.disable();

        const color = REGION_COLORS[regions.length % REGION_COLORS.length];
        currentDrawingRect = L.rectangle([startLatLng, startLatLng], {
            color: color, weight: 2, fillColor: color, fillOpacity: 0.12,
            dashArray: '6 3',
        }).addTo(map);

        map.on('mousemove', onDrawMove);
        map.on('mouseup', onDrawEnd);
    }

    function onDrawMove(e) {
        if (!currentDrawingRect || !startLatLng) return;
        currentDrawingRect.setBounds(L.latLngBounds(startLatLng, e.latlng));
    }

    function onDrawEnd(e) {
        map.off('mousemove', onDrawMove);
        map.off('mouseup', onDrawEnd);

        // Re-enable dragging so user can pan between draws
        map.dragging.enable();

        if (currentDrawingRect && startLatLng) {
            currentDrawingRect.setBounds(L.latLngBounds(startLatLng, e.latlng));
            const bounds = currentDrawingRect.getBounds();

            // Reject tiny/accidental draws (< ~20px drag)
            const startPt = map.latLngToContainerPoint(startLatLng);
            const endPt = map.latLngToContainerPoint(e.latlng);
            const dragDist = Math.sqrt(Math.pow(endPt.x - startPt.x, 2) + Math.pow(endPt.y - startPt.y, 2));

            if (dragDist < 15) {
                map.removeLayer(currentDrawingRect);
                currentDrawingRect = null;
                return; // Too small — treat as click, stay in draw mode
            }

            const color = REGION_COLORS[regions.length % REGION_COLORS.length];
            const region = {
                id: nextRegionId++,
                rect: currentDrawingRect,
                gridLayer: null,
                bounds: bounds,
                color: color,
            };
            regions.push(region);
            showOutputGrid(region);
            renderRegionList();
            updateEstimate();
            updateDrawButton();
        }
        currentDrawingRect = null;
        // Stay in draw mode — user can pan the map then draw more regions
    }

    // ======================
    // Region List
    // ======================

    function renderRegionList() {
        const container = getEl('sat-region-list');
        const items = getEl('sat-region-items');

        if (regions.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'flex';
        items.innerHTML = '';

        regions.forEach((r, idx) => {
            const item = document.createElement('div');
            item.className = 'sat-region-item';
            item.innerHTML = `
                <span class="sat-region-color" style="background:${r.color}"></span>
                <span class="sat-region-label">Region ${idx + 1}</span>
                <button class="sat-region-delete" title="Remove region">\u00D7</button>
            `;

            // Hover to highlight on map
            item.addEventListener('mouseenter', () => {
                r.rect.setStyle({ weight: 3, fillOpacity: 0.25 });
            });
            item.addEventListener('mouseleave', () => {
                r.rect.setStyle({ weight: 2, fillOpacity: 0.12 });
            });

            // Delete button
            item.querySelector('.sat-region-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                removeRegion(r.id);
            });

            items.appendChild(item);
        });
    }

    function removeRegion(regionId) {
        const idx = regions.findIndex(r => r.id === regionId);
        if (idx === -1) return;

        const r = regions[idx];
        map.removeLayer(r.rect);
        if (r.gridLayer) map.removeLayer(r.gridLayer);
        regions.splice(idx, 1);

        renderRegionList();
        updateDrawButton();

        if (regions.length > 0) {
            updateEstimate();
        } else {
            estimateData = null;
            getEl('sat-capture-info').style.display = 'none';
            getEl('btn-extract-tiles').disabled = true;
            getEl('btn-append-tiles').disabled = true;
        }
    }

    // ======================
    // Output Grid Preview
    // ======================

    function showOutputGrid(region) {
        if (region.gridLayer) { map.removeLayer(region.gridLayer); }
        region.gridLayer = L.layerGroup().addTo(map);

        const bounds = region.bounds;
        const nw = bounds.getNorthWest();
        const se = bounds.getSouthEast();
        const [xMin, yMin] = latLngToTile(nw.lat, nw.lng, currentZoom);
        const [xMax, yMax] = latLngToTile(se.lat, se.lng, currentZoom);

        const cols = xMax - xMin + 1;
        const rows = yMax - yMin + 1;
        const pw = cols * 256;
        const ph = rows * 256;

        const stride = Math.max(1, Math.floor(outputTileSize * (1 - overlapPct / 100)));

        // Snapped geographic corners via tile math
        const [nwLat, nwLng] = tileToLatLng(xMin, yMin, currentZoom);
        const [seLat, seLng] = tileToLatLng(xMax + 1, yMax + 1, currentZoom);

        function pxToLng(px) {
            return nwLng + (px / pw) * (seLng - nwLng);
        }
        function pxToLat(py) {
            const tileY = yMin + py / 256;
            const n = Math.pow(2, currentZoom);
            const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * tileY / n)));
            return latRad * 180 / Math.PI;
        }

        const lat1 = pxToLat(0);
        const lat2 = pxToLat(ph);
        const lng1 = pxToLng(0);
        const lng2 = pxToLng(pw);
        const lineStyle = { color: region.color, weight: 1, opacity: 0.4, dashArray: '3 5' };

        const vLines = Math.ceil(pw / stride) + 1;
        const hLines = Math.ceil(ph / stride) + 1;
        if (vLines + hLines > 400) return;

        for (let px = 0; px <= pw; px += stride) {
            const lng = pxToLng(px);
            L.polyline([[lat1, lng], [lat2, lng]], lineStyle).addTo(region.gridLayer);
        }
        if (pw % stride !== 0) {
            L.polyline([[lat1, lng2], [lat2, lng2]], lineStyle).addTo(region.gridLayer);
        }

        for (let py = 0; py <= ph; py += stride) {
            const lat = pxToLat(py);
            L.polyline([[lat, lng1], [lat, lng2]], lineStyle).addTo(region.gridLayer);
        }
        if (ph % stride !== 0) {
            L.polyline([[lat2, lng1], [lat2, lng2]], lineStyle).addTo(region.gridLayer);
        }
    }

    // ======================
    // Estimate
    // ======================

    async function updateEstimate() {
        if (regions.length === 0) return;

        const regionData = regions.map(r => ({
            south: r.bounds.getSouth(),
            west: r.bounds.getWest(),
            north: r.bounds.getNorth(),
            east: r.bounds.getEast(),
        }));

        try {
            estimateData = await api('api/datasets/tile-estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    regions: regionData,
                    zoom: currentZoom,
                    output_tile_size: outputTileSize,
                    overlap_pct: overlapPct,
                }),
            });
            renderEstimate(estimateData);
        } catch (e) { /* error shown by api() */ }
    }

    function renderEstimate(data) {
        getEl('sat-region-count').textContent = data.region_count;
        getEl('sat-raw-tile-count').textContent = data.raw_tile_count;
        getEl('sat-tile-count').textContent = data.output_tile_count;
        getEl('sat-area').textContent = `${data.area_km2} km\u00B2`;
        getEl('sat-resolution').textContent = `${data.resolution_m_px}m/px`;

        const extractBtn = getEl('btn-extract-tiles');
        const appendBtn = getEl('btn-append-tiles');
        const rawEl = getEl('sat-raw-tile-count');
        const warning = getEl('sat-limit-warning');

        if (data.exceeds_max) {
            extractBtn.disabled = true;
            appendBtn.disabled = true;
            rawEl.classList.add('over-limit');
            warning.style.display = 'block';
            warning.textContent = `${data.raw_tile_count} raw tiles exceeds max ${data.max_tiles}. Reduce area or zoom.`;
        } else {
            extractBtn.disabled = false;
            appendBtn.disabled = false;
            rawEl.classList.remove('over-limit');
            warning.style.display = 'none';
        }

        getEl('sat-capture-info').style.display = 'flex';

        // Show correct button based on whether we already have a dataset
        updateActionButtons();
    }

    function updateActionButtons() {
        const extractBtn = getEl('btn-extract-tiles');
        const appendBtn = getEl('btn-append-tiles');

        if (activeDatasetId && regions.length > 0) {
            // We have an existing dataset — show append button
            extractBtn.style.display = 'none';
            appendBtn.style.display = '';
        } else if (regions.length > 0) {
            // No dataset yet — show extract button
            extractBtn.style.display = '';
            appendBtn.style.display = 'none';
        } else {
            extractBtn.style.display = '';
            extractBtn.disabled = true;
            appendBtn.style.display = 'none';
        }
    }

    // ======================
    // Extraction / Append
    // ======================

    async function extractDataset(isAppend) {
        if (regions.length === 0 || !estimateData || estimateData.exceeds_max) return;

        const name = getEl('sat-dataset-name')?.value.trim() || 'Satellite Capture';
        const totalRaw = estimateData.raw_tile_count;

        const regionData = regions.map(r => ({
            south: r.bounds.getSouth(),
            west: r.bounds.getWest(),
            north: r.bounds.getNorth(),
            east: r.bounds.getEast(),
        }));

        const extractBtn = getEl('btn-extract-tiles');
        const appendBtn = getEl('btn-append-tiles');
        extractBtn.disabled = true;
        extractBtn.style.display = 'none';
        appendBtn.disabled = true;
        appendBtn.style.display = 'none';

        // Show progress bar
        const progressContainer = getEl('sat-progress');
        const progressFill = getEl('sat-progress-fill');
        const progressText = getEl('sat-progress-text');
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = isAppend
            ? `Appending 0 / ${totalRaw} tiles...`
            : `Downloading 0 / ${totalRaw} tiles...`;

        let cumulativeDownloaded = 0;
        let lastRegionDownloaded = 0;

        // Choose endpoint
        const endpoint = isAppend
            ? `api/datasets/${activeDatasetId}/append-tiles`
            : 'api/datasets/from-tiles';

        const payload = { regions: regionData, zoom: currentZoom,
            output_tile_size: outputTileSize, overlap_pct: overlapPct };
        if (!isAppend) payload.name = name;

        try {
            const resp = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let result = null;
            const regionCount = regions.length;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const msg = JSON.parse(line);
                        if (msg.type === 'progress' && msg.phase === 'download') {
                            const totalSoFar = cumulativeDownloaded + msg.downloaded;
                            lastRegionDownloaded = msg.downloaded;
                            const pct = Math.round(totalSoFar / totalRaw * 70);
                            progressFill.style.width = `${pct}%`;
                            const label = regionCount > 1 ? `Region ${msg.region + 1}: ` : '';
                            progressText.textContent = `${label}Downloading ${totalSoFar} / ${totalRaw} tiles...`;
                        } else if (msg.type === 'progress' && msg.phase === 'reslice') {
                            if (msg.status === 'stitching') {
                                cumulativeDownloaded += lastRegionDownloaded;
                                lastRegionDownloaded = 0;
                                const pct = 70 + Math.round((msg.region + 0.5) / regionCount * 30);
                                progressFill.style.width = `${pct}%`;
                                const label = regionCount > 1 ? `Region ${msg.region + 1}: ` : '';
                                progressText.textContent = `${label}Stitching & re-slicing...`;
                            } else if (msg.status === 'complete') {
                                const pct = 70 + Math.round((msg.region + 1) / regionCount * 30);
                                progressFill.style.width = `${pct}%`;
                            }
                        } else if (msg.type === 'complete') {
                            result = msg;
                        } else if (msg.type === 'error') {
                            throw new Error(msg.error);
                        }
                    } catch (parseErr) {
                        if (parseErr.message && !parseErr.message.includes('JSON')) throw parseErr;
                    }
                }
            }

            // Process remaining buffer
            if (buffer.trim()) {
                try {
                    const msg = JSON.parse(buffer);
                    if (msg.type === 'complete') result = msg;
                    else if (msg.type === 'error') throw new Error(msg.error);
                } catch (e) { /* ignore */ }
            }

            if (result) {
                progressFill.style.width = '100%';
                if (isAppend) {
                    progressText.textContent = `Appended ${result.appended} tiles (${result.image_count} total)`;
                    showToast(`Added ${result.appended} tiles to dataset (${result.image_count} total)`, 'success');
                } else {
                    progressText.textContent = `Extracted ${result.image_count} output tiles`;
                    showToast(`Dataset created: ${result.image_count} tiles at ${result.resolution_m_px || ZOOM_RESOLUTIONS[currentZoom]}m/px`, 'success');
                }

                activeDatasetId = result.dataset_id;
                pipelineState._satelliteDatasetId = result.dataset_id;
                getEl('btn-next-0').disabled = false;
                unlockStage(1);

                if (result.failed > 0) {
                    showToast(`${result.failed} tiles failed to download`, 'info');
                }

                // Clear drawn regions (they've been extracted) but keep dataset ID for future appends
                clearRegionsFromMap();
            } else {
                showToast('Extraction completed but no result received', 'error');
            }
        } catch (e) {
            showToast(e.message || 'Extraction failed', 'error');
        } finally {
            setTimeout(() => { progressContainer.style.display = 'none'; }, 2000);
            updateActionButtons();
        }
    }

    // ======================
    // Clear
    // ======================

    function clearRegionsFromMap() {
        // Remove regions from the map but preserve activeDatasetId
        regions.forEach(r => {
            map.removeLayer(r.rect);
            if (r.gridLayer) map.removeLayer(r.gridLayer);
        });
        regions = [];
        nextRegionId = 0;
        estimateData = null;
        startLatLng = null;
        renderRegionList();
        getEl('sat-capture-info').style.display = 'none';
        updateActionButtons();
    }

    function clearRegions() {
        clearRegionsFromMap();
        // Full clear also resets the active dataset
        activeDatasetId = null;
        getEl('btn-extract-tiles').disabled = true;
        getEl('btn-next-0').disabled = !pipelineState._satelliteDatasetId;
        updateActionButtons();
    }

    // ======================
    // Tile Math (mirrors server)
    // ======================

    function latLngToTile(lat, lng, zoom) {
        const n = Math.pow(2, zoom);
        const x = Math.floor((lng + 180) / 360 * n);
        const latRad = lat * Math.PI / 180;
        const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n);
        return [x, y];
    }

    function tileToLatLng(x, y, zoom) {
        const n = Math.pow(2, zoom);
        const lon = x / n * 360 - 180;
        const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n)));
        const lat = latRad * 180 / Math.PI;
        return [lat, lon];
    }

    return { init, showView };
})();
