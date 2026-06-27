# Music Analyzer

This is an university project practicum for RUPP ITE 2nd year class.

A web-based application that analyzes audio files using Essentia.js and TensorFlow.js, operating entirely within the browser.

## Features

- Analyze audio files
- Predict moods (danceability, aggressiveness, happiness)
- Display & Save results

## Why?

| Feature | Music Analyzer | Tunebat Analyzer (Free) | Tunebat Analyzer (Pro) |
|---------|--------------------------|-------------------------|------------------------|
| BPM & Key Detection | ✅ Yes | ✅ Yes | ✅ Yes |
| Mood & Energy Prediction | ✅ Yes (Machine Learning) | ❌ No | ✅ Yes |
| Save Analysis History | ✅ Yes (Local Storage) | ❌ No | ✅ Yes |
| Cost | 100% Free | Free (Ad-supported) | Paid Subscription |
| Privacy / Processing | 100% Local (Browser) | Local & Server/Cloud dependent | Local & Server/Cloud dependent |

## Technologies Used & Credits

This project is made possible thanks to the following open-source technologies and libraries:

* **[Essentia.js](https://essentia.upf.edu/essentiajs/)** - Used for core audio analysis, extracting features like BPM, key, and various other music descriptors. Essentia is an open-source library for audio and music analysis developed by the Music Technology Group (MTG) at Universitat Pompeu Fabra (UPF).
* **[TensorFlow.js](https://www.tensorflow.org/js)** - Used to run machine learning models directly in the browser to predict moods (danceability, aggressiveness, happiness) based on the audio features extracted by Essentia.js.
* **[Bootstrap 5](https://getbootstrap.com/)** - Used for the responsive layout, UI components, styling, and theme switching capabilities.
