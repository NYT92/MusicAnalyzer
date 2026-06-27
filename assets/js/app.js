const AudioContext = window.AudioContext || window.webkitAudioContext;
const audioCtx = new AudioContext();
const KEEP_PERCENTAGE = 0.15;

const MODEL_NAMES = ['mood_happy', 'danceability', 'mood_aggressive'];
const MODEL_BASE_URL = '/assets/models';
const STORAGE_KEY = 'musicanalyzer_history';

let essentia = null;
let featureExtractionWorker = null;
let models = {};
let currentAnalysis = null;

const MODEL_TAG_ORDER = {
    'mood_happy': [true, false],
    'danceability': [true, false],
    'mood_aggressive': [true, false]
};

function initEssentia() {
    return new Promise((resolve) => {
        if (typeof EssentiaWASM !== 'undefined') {
            EssentiaWASM().then((wasmModule) => {
                essentia = new wasmModule.EssentiaJS(false);
                essentia.arrayToVector = wasmModule.arrayToVector;
                console.log('Essentia.js initialized successfully');
                resolve(true);
            }).catch((err) => {
                console.error('Failed to initialize Essentia.js:', err);
                resolve(false);
            });
        } else {
            console.error('EssentiaWASM not loaded');
            resolve(false);
        }
    });
}

async function initModels() {
    try {
        if (tf.getBackend() !== 'wasm') {
            try {
                tf.wasm.setWasmPaths('./assets/lib/');
                await tf.setBackend('wasm');
                await tf.ready();
                console.log('TensorFlow.js WASM backend initialized');
            } catch (e) {
                console.warn('WASM backend failed, using default:', tf.getBackend());
            }
        }

        for (const modelName of MODEL_NAMES) {
            const modelUrl = `${MODEL_BASE_URL}/${modelName}-musicnn-msd-2/model.json`;
            console.log(`Loading model: ${modelName} from ${modelUrl}`);
            try {
                models[modelName] = new EssentiaModel.TensorflowMusiCNN(tf, modelUrl);
                await models[modelName].initialize();
                console.log(`Model ${modelName} loaded successfully`);

                const fakeFeatures = {
                    melSpectrum: getZeroMatrix(187, 96),
                    frameSize: 187,
                    melBandsSize: 96,
                    patchSize: 187
                };
                await models[modelName].predict(fakeFeatures, false);
                console.log(`Model ${modelName} warmed up`);
            } catch (err) {
                console.error(`Error loading model ${modelName}:`, err);
            }
        }
        console.log('All models initialized');
    } catch (err) {
        console.error('Error initializing models:', err);
    }
}

function getZeroMatrix(x, y) {
    const matrix = new Array(x);
    for (let f = 0; f < x; f++) {
        matrix[f] = new Array(y).fill(0);
    }
    return matrix;
}

function twoValuesAverage(arrayOfArrays) {
    const firstValues = [];
    const secondValues = [];
    arrayOfArrays.forEach((v) => {
        firstValues.push(v[0]);
        secondValues.push(v[1]);
    });
    const firstValuesAvg = firstValues.reduce((acc, val) => acc + val, 0) / firstValues.length;
    const secondValuesAvg = secondValues.reduce((acc, val) => acc + val, 0) / secondValues.length;
    return [firstValuesAvg, secondValuesAvg];
}

function preprocess(audioBuffer) {
    if (audioBuffer instanceof AudioBuffer) {
        const mono = monomix(audioBuffer);
        return downsampleArray(mono, audioBuffer.sampleRate, 16000);
    }
    throw new TypeError("Input to audio preprocessing is not of type AudioBuffer");
}

function monomix(buffer) {
    if (buffer.numberOfChannels > 1) {
        const leftCh = buffer.getChannelData(0);
        const rightCh = buffer.getChannelData(1);
        return leftCh.map((sample, i) => 0.5 * (sample + rightCh[i]));
    }
    return buffer.getChannelData(0);
}

function downsampleArray(audioIn, sampleRateIn, sampleRateOut) {
    if (sampleRateOut === sampleRateIn) return audioIn;
    const sampleRateRatio = sampleRateIn / sampleRateOut;
    const newLength = Math.round(audioIn.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetAudioIn = 0;

    while (offsetResult < result.length) {
        const nextOffsetAudioIn = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetAudioIn; i < nextOffsetAudioIn && i < audioIn.length; i++) {
            accum += audioIn[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetAudioIn = nextOffsetAudioIn;
    }
    return result;
}

function shortenAudio(audioIn, keepRatio = 0.5, trim = false) {
    if (keepRatio < 0.15) keepRatio = 0.15;
    else if (keepRatio > 0.66) keepRatio = 0.66;

    if (trim) {
        const discardSamples = Math.floor(0.1 * audioIn.length);
        audioIn = audioIn.subarray(discardSamples, audioIn.length - discardSamples);
    }

    const ratioSampleLength = Math.ceil(audioIn.length * keepRatio);
    const patchSampleLength = 187 * 256;
    const numPatchesToKeep = Math.max(1, Math.ceil(ratioSampleLength / patchSampleLength));
    const skipSize = numPatchesToKeep > 1
        ? Math.floor((audioIn.length - ratioSampleLength) / (numPatchesToKeep - 1))
        : 0;

    const audioOut = [];
    let startIndex = 0;
    for (let i = 0; i < numPatchesToKeep; i++) {
        const endIndex = startIndex + patchSampleLength;
        const chunk = audioIn.slice(startIndex, endIndex);
        audioOut.push(...chunk);
        startIndex = endIndex + skipSize;
    }
    return Float32Array.from(audioOut);
}

function computeKeyBPM(audioSignal) {
    const vectorSignal = essentia.arrayToVector(audioSignal);
    const keyData = essentia.KeyExtractor(vectorSignal, true, 4096, 4096, 12, 3500, 60, 25, 0.2, 'bgate', 16000, 0.0001, 440, 'cosine', 'hann');
    const bpm = essentia.PercivalBpmEstimator(vectorSignal, 1024, 2048, 128, 128, 210, 50, 16000).bpm;
    return {
        keyData,
        bpm,
        tempo: classifyTempo(bpm)
    };
}

function classifyTempo(bpm) {
    if (bpm < 80) return { category: 'Slow', confidence: Math.min(1, (80 - bpm) / 40 + 0.5) };
    if (bpm < 100) return { category: 'Medium-Slow', confidence: 1 - Math.abs(bpm - 90) / 20 };
    if (bpm < 120) return { category: 'Medium', confidence: 1 - Math.abs(bpm - 110) / 20 };
    if (bpm < 140) return { category: 'Medium-Fast', confidence: 1 - Math.abs(bpm - 130) / 20 };
    if (bpm < 160) return { category: 'Fast', confidence: 1 - Math.abs(bpm - 150) / 20 };
    return { category: 'Very Fast', confidence: Math.min(1, (bpm - 140) / 40) };
}

function getCamelotKey(key, scale) {
    const camelotMap = {
        'A': { 'minor': '8A', 'major': '11B' },
        'A#': { 'minor': '3A', 'major': '6B' },
        'Bb': { 'minor': '3A', 'major': '6B' },
        'B': { 'minor': '10A', 'major': '1B' },
        'C': { 'minor': '5A', 'major': '8B' },
        'C#': { 'minor': '12A', 'major': '3B' },
        'Db': { 'minor': '12A', 'major': '3B' },
        'D': { 'minor': '7A', 'major': '10B' },
        'D#': { 'minor': '2A', 'major': '5B' },
        'Eb': { 'minor': '2A', 'major': '5B' },
        'E': { 'minor': '9A', 'major': '12B' },
        'F': { 'minor': '4A', 'major': '7B' },
        'F#': { 'minor': '11A', 'major': '2B' },
        'Gb': { 'minor': '11A', 'major': '2B' },
        'G': { 'minor': '6A', 'major': '9B' },
        'G#': { 'minor': '1A', 'major': '4B' },
        'Ab': { 'minor': '1A', 'major': '4B' }
    };
    const scaleLower = scale.toLowerCase();
    if (camelotMap[key] && camelotMap[key][scaleLower]) return camelotMap[key][scaleLower];
    return '-';
}

function createFeatureExtractionWorker() {
    featureExtractionWorker = new Worker('./assets/js/feature-worker.js');
    featureExtractionWorker.onmessage = async function (msg) {
        if (msg.data.features) {
            const predictions = {};
            for (const modelName of MODEL_NAMES) {
                if (models[modelName]) {
                    try {
                        const inferenceStart = Date.now();
                        const modelPredictions = await models[modelName].predict(msg.data.features, true);
                        const summarizedPredictions = twoValuesAverage(modelPredictions);
                        const tagOrder = MODEL_TAG_ORDER[modelName];
                        predictions[modelName] = summarizedPredictions.filter((_, i) => tagOrder[i])[0];
                        console.log(`${modelName}: Inference took: ${Date.now() - inferenceStart}ms`);
                    } catch (err) {
                        console.error(`Error running inference for ${modelName}:`, err);
                    }
                }
            }
            console.log(predictions);
            currentAnalysis.moods = predictions;
            updateUI(predictions);
            toggleLoader(false);
        }
        featureExtractionWorker.terminate();
    };
}

function updateUI(predictions) {
    updateProgressBar('danceability-value', predictions.danceability, 'danceability-pct');
    updateProgressBar('aggressive-value', predictions.mood_aggressive, 'aggressive-pct');
    updateProgressBar('happiness-value', predictions.mood_happy, 'happiness-pct');
}

function updateProgressBar(barId, value, pctId) {
    if (value === undefined) return;
    const pct = Math.round(value * 100);
    const bar = document.getElementById(barId);
    if (bar) {
        bar.style.width = pct + '%';
        bar.setAttribute('aria-valuenow', pct);
    }
    const pctEl = document.getElementById(pctId);
    if (pctEl) pctEl.textContent = pct + '%';
}

function updateValueBoxes(analysis) {
    const bpmElement = document.getElementById('bpm');
    if (bpmElement && analysis.bpm) {
        bpmElement.textContent = Math.round(analysis.bpm);
        setComponentLoading('bpm', false);
    }
    const keysElement = document.getElementById('keys');
    if (keysElement && analysis.keyData) {
        keysElement.textContent = analysis.keyData.key + ' ' + analysis.keyData.scale;
        setComponentLoading('keys', false);
    }
    const camelotElement = document.getElementById('camelot');
    if (camelotElement && analysis.keyData) {
        const camelot = getCamelotKey(analysis.keyData.key, analysis.keyData.scale);
        camelotElement.textContent = camelot;
        setComponentLoading('camelot', false);
    }
    const tempoCategoryElement = document.getElementById('tempo-category');
    const tempoConfidenceElement = document.getElementById('tempo-confidence');
    if (tempoCategoryElement && analysis.tempo) {
        tempoCategoryElement.textContent = analysis.tempo.category;
    }
    if (tempoConfidenceElement && analysis.tempo) {
        tempoConfidenceElement.textContent = '(' + Math.round(analysis.tempo.confidence * 100) + '% confidence)';
    }
}

function setComponentLoading(elementId, isLoading) {
    const container = document.getElementById(elementId + '-container');
    if (!container) return;
    const loadingIndicator = container.querySelector('.loading-indicator');
    const valueDisplay = container.querySelector('.value-display');
    if (isLoading) {
        if (loadingIndicator) loadingIndicator.style.display = 'inline-block';
        if (valueDisplay) valueDisplay.style.display = 'none';
    } else {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (valueDisplay) valueDisplay.style.display = 'block';
    }
}

function setAllComponentsLoading(isLoading) {
    ['bpm', 'keys', 'camelot', 'duration'].forEach(id => setComponentLoading(id, isLoading));
}

function toggleLoader(show) {
    const loader = document.getElementById('loading-container');
    if (loader) loader.classList.toggle('d-none', !show);
}

function showResults() {
    document.getElementById('results-section').classList.remove('d-none');
}

function resetResults() {
    document.getElementById('results-section').classList.add('d-none');
    ['bpm', 'keys', 'camelot', 'duration'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '-';
    });
    ['tempo-category', 'tempo-confidence'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '';
    });
    ['danceability-value', 'aggressive-value', 'happiness-value'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.style.width = '0%'; el.setAttribute('aria-valuenow', '0'); }
    });
    ['danceability-pct', 'aggressive-pct', 'happiness-pct'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '0%';
    });
}

// localStorage helpers
function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch {
        return [];
    }
}

function saveToHistory(entry) {
    const history = getHistory();
    const existingIdx = history.findIndex(h => h.fileName === entry.fileName);
    if (existingIdx !== -1) {
        history[existingIdx] = entry;
    } else {
        history.unshift(entry);
    }
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        console.error('Failed to save to localStorage:', e);
    }
    renderHistory();
}

function deleteFromHistory(fileName) {
    const history = getHistory().filter(h => h.fileName !== fileName);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    localStorage.removeItem(STORAGE_KEY);
    renderHistory();
}

function renderHistory() {
    const history = getHistory();
    const section = document.getElementById('history-section');
    const tbody = document.getElementById('history-body');

    if (history.length === 0) {
        section.classList.add('d-none');
        return;
    }

    section.classList.remove('d-none');
    tbody.innerHTML = '';

    history.forEach(entry => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.innerHTML = `
            <td class="text-truncate" style="max-width:200px" title="${escapeHtml(entry.fileName)}">${escapeHtml(entry.fileName.replace(/\.[^.]+$/, ''))}</td>
            <td>${entry.bpm != null ? Math.round(entry.bpm) : '-'}</td>
            <td>${entry.key || '-'}</td>
            <td>${entry.camelot || '-'}</td>
            <td>${entry.danceability != null ? Math.round(entry.danceability * 100) + '%' : '-'}</td>
            <td>${entry.happiness != null ? Math.round(entry.happiness * 100) + '%' : '-'}</td>
            <td>${entry.aggressiveness != null ? Math.round(entry.aggressiveness * 100) + '%' : '-'}</td>
            <td>
                <button class="btn btn-sm btn-outline-danger border-0 delete-entry" data-name="${escapeAttr(entry.fileName)}" title="Delete">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5m3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0z"/>
                        <path d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H5.5l1-1h3l1 1h2.5a1 1 0 0 1 1 1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4zM2.5 3V2h11v1z"/>
                    </svg>
                </button>
            </td>
        `;
        tr.addEventListener('click', (e) => {
            if (e.target.closest('.delete-entry')) return;
            loadFromHistory(entry);
        });
        tr.querySelector('.delete-entry').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteFromHistory(entry.fileName);
        });
        tbody.appendChild(tr);
    });
}

function loadFromHistory(entry) {
    resetResults();
    currentAnalysis = {
        fileName: entry.fileName,
        bpm: entry.bpm,
        key: entry.key,
        camelot: entry.camelot,
        duration: entry.duration,
        tempo: entry.tempo,
        moods: {
            danceability: entry.danceability,
            mood_happy: entry.happiness,
            mood_aggressive: entry.aggressiveness
        }
    };

    document.getElementById('result-song-name').textContent = entry.fileName.replace(/\.[^.]+$/, '');
    document.getElementById('bpm').textContent = Math.round(entry.bpm);
    document.getElementById('keys').textContent = entry.key;
    document.getElementById('camelot').textContent = entry.camelot;
    document.getElementById('duration').textContent = entry.duration;

    const tempoEl = document.getElementById('tempo-category');
    const tempoConfEl = document.getElementById('tempo-confidence');
    if (entry.tempo) {
        tempoEl.textContent = entry.tempo.category;
        tempoConfEl.textContent = '(' + Math.round(entry.tempo.confidence * 100) + '% confidence)';
    }

    updateUI(currentAnalysis.moods);

    ['bpm', 'keys', 'camelot', 'duration'].forEach(id => setComponentLoading(id, false));
    showResults();
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function processAudioFile(file) {
    try {
        currentAnalysis = {
            fileName: file.name,
            bpm: null,
            key: null,
            camelot: null,
            duration: null,
            tempo: null,
            moods: {}
        };

        resetResults();
        showResults();
        document.getElementById('result-song-name').textContent = file.name.replace(/\.[^.]+$/, '');
        toggleLoader(true);
        setAllComponentsLoading(true);

        const arrayBuffer = await file.arrayBuffer();
        await audioCtx.resume();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const preprocessedAudio = preprocess(audioBuffer);
        await audioCtx.suspend();

        const durationMinutes = Math.floor(audioBuffer.duration / 60);
        const durationSeconds = Math.floor(audioBuffer.duration % 60);
        const durationStr = durationMinutes + ':' + durationSeconds.toString().padStart(2, '0');

        currentAnalysis.duration = durationStr;

        const durationElement = document.getElementById('duration');
        if (durationElement) {
            durationElement.textContent = durationStr;
            setComponentLoading('duration', false);
        }

        let analysis = null;
        if (essentia) {
            analysis = computeKeyBPM(preprocessedAudio);
            updateValueBoxes(analysis);

            currentAnalysis.bpm = analysis.bpm;
            currentAnalysis.key = analysis.keyData.key + ' ' + analysis.keyData.scale;
            currentAnalysis.camelot = getCamelotKey(analysis.keyData.key, analysis.keyData.scale);
            currentAnalysis.tempo = analysis.tempo;
        }

        const shortenedAudio = shortenAudio(preprocessedAudio, KEEP_PERCENTAGE, true);
        createFeatureExtractionWorker();

        const audioCopy = shortenedAudio.slice();
        featureExtractionWorker.postMessage({ audio: audioCopy.buffer }, [audioCopy.buffer]);

    } catch (error) {
        console.error('Error processing audio:', error);
        toggleLoader(false);
        alert('Error processing audio file. Please try again.');
    }
}

function saveCurrentResult() {
    if (!currentAnalysis) return;
    saveToHistory({
        fileName: currentAnalysis.fileName,
        bpm: currentAnalysis.bpm,
        key: currentAnalysis.key,
        camelot: currentAnalysis.camelot,
        duration: currentAnalysis.duration,
        tempo: currentAnalysis.tempo,
        danceability: currentAnalysis.moods.danceability,
        happiness: currentAnalysis.moods.mood_happy,
        aggressiveness: currentAnalysis.moods.mood_aggressive,
        savedAt: new Date().toISOString()
    });
}

async function init() {
    await initEssentia();
    await initModels();
    renderHistory();

    const analyzeBtn = document.getElementById('analyze-btn');
    const audioInput = document.getElementById('audio-input');
    const dropZone = document.getElementById('drop-zone');
    const fileNameDisplay = document.getElementById('file-name-display');
    const dropLabel = document.getElementById('drop-label');
    const saveBtn = document.getElementById('save-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    dropZone.addEventListener('click', () => audioInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('audio/')) {
            audioInput.files = files;
            onFileSelected(files[0]);
        }
    });

    audioInput.addEventListener('change', () => {
        if (audioInput.files.length > 0) {
            onFileSelected(audioInput.files[0]);
        }
    });

    function onFileSelected(file) {
        analyzeBtn.disabled = false;
        dropLabel.textContent = file.name;
        fileNameDisplay.textContent = formatFileSize(file.size);
    }

    analyzeBtn.addEventListener('click', () => {
        if (audioInput && audioInput.files.length > 0) {
            processAudioFile(audioInput.files[0]);
        }
    });

    saveBtn.addEventListener('click', saveCurrentResult);

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Clear all saved analysis history?')) {
            clearHistory();
        }
    });

    toggleLoader(false);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

document.addEventListener('DOMContentLoaded', function () {
    init();
});
