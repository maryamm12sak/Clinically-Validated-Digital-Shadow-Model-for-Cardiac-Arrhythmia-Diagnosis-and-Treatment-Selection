# Commit ID map

Before this branch was first pushed, the commit messages were cleaned: two trailer lines were removed
from older commits. File contents (trees), authors, committers and all dates are **unchanged**.
Only the commit IDs changed, because a commit's ID covers its message.

**Pre-registration commit:** committed at 16:55:19 UTC on 28 Oct 2026, 17 seconds before fold 10 was
opened for the one-time test at 16:55:36 UTC. It was `14486d8` and is now **`b2a060e`**.

**Documents with old IDs:** some files were written before the rewrite and still quote the old IDs.
Use the table below to translate them:
- `ml/deploy/MODEL_CARD.md` and `ml/deploy/manifest.json` (pre-registration `14486d8`; training commit `003acbf`)
- `ml/export_deploy.py` (`14486d8`)
- `ml/results/final/preregistration.json` (`git_head` `67b7916`)

**Why those files were left unchanged:** they are part of the hash-checked, pre-registered record or the
model package.

| Old ID | New ID | Subject |
|---|---|---|
| ce4309c0b85cc3ae51769880dc6fbf541dd74fdd | 7988d062dd921393f199a8232fc4cf23d09e7035 | Add PTB-XL data preparation modules and tests |
| 19bea8d7d4b96a0a6b2aa390a5cee44ef845d144 | d326e91d45f44a4e8ffc99e83191d5d05bb3260f | Add classical baseline benchmark and evaluation |
| 1d2b1712dd2be96c4855c3e86ec1bc91f6087077 | abd33f2ab42405a3898df89a4a2cd0b7c2459947 | Add deep-learning training loop with CNN and ResNet baselines |
| 67b7916a1836af424333140a2c8912cd0c84a4c7 | efcf3a220a38a55fb0f873dd64a8e14638f2be11 | Add InceptionTime, CNN-LSTM, ensembling and calibration |
| 14486d876d5337c1c5f46ddbd4a1d8ad6c4da9e0 | b2a060efd70621f267567887269b7d6aa5e61ae9 | Pre-register final candidates for fold-10 evaluation |
| 003acbf3cdd25d83b13dc4f4951f14652e6b8cb4 | 982d5ef1a2e7e5e33b45a68120073ca280533ac5 | Add locked fold-10 evaluation and fix in-place normalisation |
| f46e0b0ba505cd34e65db8089e32c5fedb666e59 | 31457017256a2eba98150f9a59b5f008bba91db0 | Add deployable model package for backend integration |
| ef5ef270c09414fe8c15386f24dcb35acbd4c962 | c7d6191f94f44e955b45253db26282421f8ffb21 | Integrate trained ResNet1d model into backend predictor |
| ca03f6189e5f52715ccc503d2d903b071daec38d | 0655476fafad6074db2763015fe9937bf0f83c78 | Track small ML result files and set line-ending rules |
| aa38d3f502ebf991056febb7ada98a825571afa5 | e0016c12a239339b9af63935c9752a5480a6434e | Add one-click run_app.bat launcher with first-run setup |
| 443b0983ad6fefe89d5ef38af6032bc676866ba3 | 4495e92ee505353d787900767b12e1d5fc250bc7 | Add project README with setup, model summary and limits |
| f38c7842febfb77f2b6790d5f9de89e7dd0e60aa | 88f420a10bce3d7798a6c4e1105f1cec0c59c2ce | Use repo-relative PTB-XL paths in benchmark config |
| 13033a197b5a7059f46e1b38465d20e075b934ad | f6d17dc216f7cec2c28de789b026703357664975 | Make the launcher's health wait work without keyboard input |

Note: `ml/configs/benchmark.yaml` is one of the pre-registered, hash-checked files. After the fold-10 test,
its three data paths were changed to repo-relative paths (commit `88f420a`), so a fresh hash check of that
file against `preregistration.json` will now report a difference. No model, threshold, calibrator or
evaluation setting was changed. The pre-registered version is in commit `b2a060e`.
