# ▶️ Step-by-step guide: running the project in Google Colab

You only need a **Google account** and a browser. No installation is needed on your laptop.

---

## Part 1: Notebook 1 (Phase 1: dataset & preprocessing). About 10 minutes.

1. Open the project on GitHub: `github.com/yashixb-cyber/plant-disease-detection-cnn`
2. In the README table, click the **"Open in Colab"** badge next to **Notebook 1**.
3. At the top, click **"Copy to Drive"**. This makes your own editable copy. (If you don't see it: `File → Save a copy in Drive`.)
4. Turn on the free GPU: **`Runtime → Change runtime type`** → under *Hardware accelerator* choose **T4 GPU** → **Save**.
5. Run everything: **`Runtime → Run all`**.
6. A pop-up asks to **connect to Google Drive**. Click **Connect to Google Drive**, choose your account, and click **Allow** (or **Continue**). This is where results get saved.
7. Wait. The download step prints `Total images : 54305` after 2–5 minutes.
8. Scroll through: you'll see the class-distribution chart, sample leaves, the split table and augmented images.

✅ **Check:** In Google Drive there is now a folder `My Drive/plant-disease-project/` with `models/class_names.json` and `results/` (split files and figures).

---

## Part 2: Notebook 2 (Phase 2: training & evaluation). About 45–75 minutes.

1. Open **Notebook 2** with its Colab badge and **Copy to Drive** again.
2. **`Runtime → Change runtime type → T4 GPU`** → Save.
3. **`Runtime → Run all`** and allow Drive access again.
4. Training starts. You'll see lines like:
   ```
   Epoch 3/25
   1188/1188 ━━━━━━━━━━━━━━━━━━━━ 95s  - accuracy: 0.8812 - loss: 0.3790 - val_accuracy: 0.9301 - val_loss: 0.2154
   ```
   - `accuracy` = score on training images, `val_accuracy` = score on validation images (both should go up)
   - `loss` = how wrong the model is (should go down)
5. **Keep the browser tab open** and your laptop awake. Free Colab disconnects if the tab sits idle for a long time.
6. When it finishes, scroll down for the test results, confusion matrices and the comparison table.

### 🔌 If Colab disconnects in the middle
No problem. The best model so far is saved to Drive after every epoch. Just do **`Runtime → Run all`** again:
- a model that **finished** is loaded instead of retrained
- a model that was **interrupted** is retrained from the start

### ⚠️ "You cannot currently connect to a GPU due to usage limits"
Free Colab has a daily GPU limit. Wait a few hours, or try another teammate's Google account (they'll need to run Notebook 1 first, or you can share the `plant-disease-project` folder with them).

✅ **Check:** `My Drive/plant-disease-project/models/` now contains `basic_cnn.keras` and `mobilenetv2.keras`, and `results/` contains `model_comparison.csv`.

---

## Part 3: Notebook 3 (Phase 3: web app & field test). About 10 minutes.

1. Open **Notebook 3**, **Copy to Drive**, select **T4 GPU**.
2. Run the cells **from top to bottom** (use `Shift + Enter`, or `Runtime → Run all`).
3. After the Gradio cell, a link appears:
   `Running on public URL: https://xxxxxxxx.gradio.live`
   Open it in your browser **or on your phone** and upload a leaf photo. 🎉
4. Continue running the cells below for the **real-world field test** (PlantDoc). It shows how accuracy drops on farm photos.

> The app link only works while the notebook is running. For the final demo, open Notebook 3 a few minutes before and run it.

---

## Part 4: Put the results into GitHub (after training)

1. In Google Drive, open `plant-disease-project/results/`, select everything, and **Download**.
2. On GitHub, open the repo → `results/` folder → **Add file → Upload files** → drag in the CSV files and the `figures` folder → **Commit changes**.
3. Fill in the **Results** table in `README.md` (GitHub → click the file → ✏️ pencil icon → edit → Commit).

Or just send the CSV files to Claude and ask it to update the repo.

> The `.keras` model files are **not** uploaded to GitHub on purpose (large binary files). They stay in Drive.

---

## 🆘 Common problems

| Problem | Fix |
|---|---|
| `MessageError: credential propagation was unsuccessful` | Google Drive permission didn't complete. `Runtime → Restart session`, run the Setup cell again, choose the **same account** as Colab, and on the permission screen tick **Select all** before *Continue*. Signed into several Google accounts? Use an Incognito window. (The notebook now continues without Drive if this fails, but then results are lost when the session ends.) |
| `GPU available: []` | You forgot to select the T4 GPU: `Runtime → Change runtime type`, then *Run all* again |
| Download step very slow / fails | Run that cell again. If GitHub is blocked on your network, try a different network or a phone hotspot |
| `No trained models found` in Notebook 3 | Run Notebook 2 completely first, with the same Google account |
| `ResourceExhaustedError` / out of memory | In the setup cell change `BATCH_SIZE = 32` to `16`, then *Run all* |
| Any other red error | Copy the full error message and send it to Claude |
