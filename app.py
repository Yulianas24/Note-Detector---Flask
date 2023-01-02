from flask import Flask, render_template, request
import joblib
import librosa
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])

def note_detection():
    
    def pitch(signal):
        fft = librosa.stft(signal, center = False, n_fft=1024)
        energi = abs(fft)**2
        data = pd.DataFrame(energi[:70])
        index = data.max().idxmax()
        formatted_data = map(lambda x: format(x, '.2f'), data[index])
        data =  list(formatted_data)
        return data

    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        source = request.files['audio_file']

        audio, sample_rate = librosa.load(source)
        audio, _ = librosa.effects.trim(audio, top_db=4)

        # Detect onsets
        onsets = librosa.onset.onset_detect(audio, sr=sample_rate)
        # Convert onsets to time in seconds
        onset_times = librosa.frames_to_time(onsets, sr=sample_rate)

        splits = []
        for i, onset_time in enumerate(onset_times):
            if i == 0:
                # Split from the beginning of the audio
                splits.append(audio[:int(sample_rate * onset_time)])
            elif i == len(onset_times) - 1:
                # Split from the previous onset to the end of the audio
                splits.append(audio[int(sample_rate * onset_times[i-1]):])
            else:
                # Split from the previous onset to the current onset
                splits.append(audio[int(sample_rate * onset_times[i-1]):int(sample_rate * onset_time)])

        datas = []
        for i, split in enumerate(splits):
            datas.append(pitch(split))
        model = joblib.load('model/model.pkl')
        result = model.predict(datas)
        return render_template('index.html', result = result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)