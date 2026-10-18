# models/ - the error model of this Space

The Space uses **`models/squat_model/`**: the export of `research/notebooks/04_final_model.ipynb`
(`research/results/04_final/export/`) with

| File | |
|---|---|
| `model.pt` | the trained weights |
| `config.json` | architecture, classes, feature version, training date, CV score |
| `norm_stats.json` | the input normalisation of the training data |
| `model_card.md` | (optional) description |

The same files as **`models/squat_model.zip`** work too (at the top of the zip or in one folder inside it).

## Change the model

1. Rename the old one, e.g. `squat_model` -> `squat_model_2026-10-18` (or delete it).
2. Upload the new export as `models/squat_model/` (the three files) or as `models/squat_model.zip`.
3. The Space restarts by itself after the upload and uses the new model. The line **Error model** at the top of the
   page shows which one is active (name, training date, CV score, folder).

Without a `squat_model`, the Space takes the newest model in `models/` (by the `created` date in its `config.json`).
Old models can stay in the folder; they are not used.

A new model must come from the same `research/squat_research` code as the Space's copy (the same
`feature_version` in `config.json`). If you changed the features or the model code, upload `research/` too; when the
versions differ, the page says so and shows the reps without errors instead of wrong ones. New classes need a
name / colour / fixes in `final_product/product_content.json` and a painted template in `saliency_maps/`.
