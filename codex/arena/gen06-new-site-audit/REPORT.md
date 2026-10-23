# GEN06 new site counterfactual audit

Fixed GEN06 SHA256 `123467a930e0598c464ae7694800ad565aeac71680b88dafc5f014bfe7cf50ac`; exact MCTS100, RNG seed 0 per query, one process / one ORT thread. GEN05 choices reused from immutable evidence. Historical source model UNKNOWN. All saved observations, legal/admissible masks and relevant witnesses independently verified against the engine. No strength games or training.

| Case | Actor / user actor | GEN05 saved | GEN06 first | Actual GEN06 second | Certificate / completion |
|---|---|---|---|---|---|
| 14878/36 | MAGENTA / True | F3->G3 | F3->G2 | n/a | PASS; avoids checked terminal loss |
| 14621/63 | MAGENTA / True | F7->F6 | F7->F6 | F6->F7 | FAIL; allows enemy terminal win |
| 16056/36 | MAGENTA / True | E5->D4 | E5->D4 | n/a | FAIL; allows enemy terminal win |
| 14009/33 | CYAN / True | B2->B3 | B2->B3 | B3->B4 | FAIL; allows enemy terminal win |
| 13352/34 | CYAN / True | C7->E5 | C7->E5 | n/a | FAIL; allows enemy terminal win |
| 15116/31 | MAGENTA / False | F6->E5 | F6->E5 | E5->F4 | FAIL; allows enemy terminal win |
| 15116/35 | MAGENTA / False | G5->F5 | G5->F5 | F5->G6 | FAIL; allows enemy terminal win |
| 15116/33 | CYAN / True | C6->C3 | C6->C3 | C3->C6 | FAIL; avoids checked terminal loss |

Only six independent games: the three 15116 positions share one game group. Defensive certificates are bounded to this turn plus the next enemy turn. These queries do not identify the model that played the website games or establish overall strength.


Remaining own-loss corrections: 14621/63 (Kernel E6->D6/D7/D5 then C6/C6/C5), 16056/36 (E5->E4/D5/D6/F4), 14009/33 (B2->C2/C3 then D2), 13352/34 (seven Ice retreats including C7->C6/C5/D6). Opponent defenses 15116/31 and /35 remain missed. Our immediate win at 15116/33 remains missed: C6->F6 then F6->G5 wins by Kernel capture; its certificate line replayed successfully. No exact winning start was selected by the model.

Raw numeric scan of all 4105 observed states at B1/8/16 passed; see NUMERIC_REPORT.md and numeric-summary.json.
