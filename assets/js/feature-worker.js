// Feature extraction worker for Music Analyzer
// Uses Essentia.js WASM module (worker-compatible version)

importScripts('../lib/essentia.js-model.umd.js');
importScripts('../lib/essentia-wasm.module.js');

const EssentiaWASM = Module;

const extractor = new EssentiaModel.EssentiaTFInputExtractor(EssentiaWASM, 'musicnn', false);

function computeFeatures(audioData) {
    const featuresStart = Date.now();

    const features = extractor.computeFrameWise(audioData, 256);

    console.info(`Feature extraction took: ${Date.now() - featuresStart}ms`);

    postMessage({ features: features });
}

onmessage = function (e) {
    if (e.data.audio) {
        const audio = new Float32Array(e.data.audio);
        computeFeatures(audio);
    }
};
