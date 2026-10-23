# COLONY: ONE — Research Dossier pentru Astra și Opus
**Data:** 23 octombrie 2026. **Scop:** cel mai puternic model ONNX pentru Level 4, nu cel mai sofisticat pipeline academic.
**Cuprins:** 106 lucrări și resurse distincte; dintre ele 101 lucrări/articole/referințe academice și 5 documentații/coduri tehnice. Numerele includ preprinturi care nu trebuie tratate automat ca lucrări peer-reviewed.

## 0. SPECIFICAȚII REALE — verificate în pachetele jocului

- **Input:** `(batch,16,8,8)` float32; 6 tipuri proprii + 6 adverse: Ping, Daemon, Exploit, Trace, Ice, Kernel; alte 4 plane: locked, actions_left, firewall urgency, actual side.
- **Output:** `policy (batch,4096)` float32 raw logits pe `(from_square,to_square)` și `value (batch,)` float32 în `[-1,1]` din perspectiva echipei care mută.
- **Depunere:** un singur fișier ONNX <= 5 MiB; opset 13-18 și doar operatori din lista permisă; batch dinamic; fără external weights.
- **Referee:** 100 MCTS simulations/acțiune în Level 4, PUCT fix, evaluare ONNX Runtime CPU, batch 8, zgomot/temperatură în primele 8 acțiuni ale meciului.
- **Reguli critice:** câte **două acțiuni consecutive per echipă**; schimbare de perspectivă numai la schimbarea turei; Kernel protection / sacrificiu de gardă; shrink firewall la rundele 8/16/24/26; Magenta câștigă tie-break. Folosește `level-4_scene.json`, nu scena implicită Level 3.
- **Notă:** Level 4 include exact piesele Trace, Daemon și Ice; nu implementa strategia după reguli simplificate din Level 1/2.

## 1. Verdict preliminar: prioritățile cu cel mai bun raport câștig/timp

1. **Validare și benchmark înainte de antrenare:** ONNX bun din punct de vedere legal, latency batch 1/8/16 și meciuri pe scena Level 4; evaluatorul ONNX poate consuma o fracție majoră din limita CPU.
2. **Teacher de tactici exacte:** enumerate one/full-turn opponent responses, kernel attack/guard sacrifice, firewall terminal, proof states. Înveți pozițiile care schimbă outcome-ul, nu doar materialul.
3. **Gumbel AlphaZero sau MCTS deep doar OFFLINE**: proporții între acțiuni legale, Q/sau root win value estimate, număr de vizite, certitudini; referee PUCT rămâne nemodificabil.
4. **Curriculum endgame + prioritizare regret/uncertainty + replay** (KataGo, Go-Exploit, RGSC, NeurIPS 2025): training density pe deciziile grele din distribuția adevărată a meciurilor.
5. **Arhitectură ablation**: 3-6 residual blocks de 32-64 canale + 2 heads vs tiny square-token attention/bilinear `(64×64)` din Chessformer; atenție la compararea **ms/eval pe CPU**, nu doar FLOPs.
6. **Iterație rapidă și promoție strictă:** re-label eșecuri ale modelului, antrenare scurtă; upload numai după jocuri paired și check valid; succesul este win rate la 100 sim, nu accuracy în train.

## 2. Bibliografia adnotată

### Fundamente (25)

- **A · 2016 · [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://doi.org/10.1038/nature16961)** — Valoare + politică + căutare; justificarea arhitecturii policy–value; nu copia complexitatea.
- **A · 2017 · [Mastering the Game of Go without Human Knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)** — Baza self-play + MCTS + loss policy/value, dar metoda brută are cost prea mare pentru competiție.
- **A · 2018 · [A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play (AlphaZero)](https://doi.org/10.1126/science.aar6404)** — Protocolul standard pentru antrenare search→policy/value și comparații ablation.
- **A · 2017 · [Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration)](https://arxiv.org/abs/1705.08439)** — Distilează teacher-ul de căutare într-o politică student; fundamental pentru pipeline-ul nostru.
- **A · 2019 · [Accelerating Self-Play Learning in Go (KataGo)](https://arxiv.org/abs/1902.10565)** — Playout cap randomization, pruning policy target, auxiliary heads, global pooling; optimizări pentru eficiența datelor.
- **A · 2022 · [Policy Improvement by Planning with Gumbel (Gumbel AlphaZero)](https://openreview.net/pdf?id=bERaNdoegnO)** — Search offline cu buget redus; target-uri de politică mai bine definite; NU putem modifica PUCT-ul serverului.
- **B · 2020 · [Monte-Carlo Tree Search as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509)** — Formalizează search ca îmbunătățire de politică; relevant pentru construcția soft-targeturilor.
- **B · 2019 · [Policy Gradient Search: Online Planning and Expert Iteration without Search Trees](https://arxiv.org/abs/1904.03646)** — Optimizare alternativă offline, numai dacă teacher-ul MCTS clasic nu oferă suficient randament.
- **B · 2023 · [MiniZero: Comparative Analysis of AlphaZero and MuZero on Go, Othello, and Atari Games](https://arxiv.org/abs/2310.11305)** — Benchmark și cod de referință pentru implementări compacte, batching, numărul de simulări și rețele.
- **B · 2026 · [Recursive Monte-Carlo Tree Search](https://proceedings.mlr.press/v306/howard26a.html)** — Teacher offline cu căutare mai puternică; verifică integrarea și costul real.
- **B · 2024 · [Speculative Monte-Carlo Tree Search](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a19940b01b77b6acd41ff8b32b334e7c-Abstract-Conference.html)** — Parallelizare speculativă a evaluatorului; investiție justificată doar după profiling.
- **B · 2019 · [Multiple Policy Value Monte Carlo Tree Search](https://www.ijcai.org/proceedings/2019/653)** — Combină rețele de capacități diferite în teacher-ul offline, nu în referee.
- **B · 2025 · [An Efficient Node Selection Policy for Monte Carlo Tree Search with Neural Networks](https://doi.org/10.1287/ijoc.2023.0307)** — Selecție de noduri la antrenare/teacher; PUCT în producție este fix.
- **B · 2020 · [Warm-Start AlphaZero Self-Play Search Enhancements](https://arxiv.org/abs/2004.12357)** — Folosește tactici și RAVE pentru bootstrap când modelul este slab; relevant pentru timp scurt.
- **B · 2020 · [Accelerating and Improving AlphaZero Using Population Based Training](https://arxiv.org/abs/2003.06212)** — Tunează hiperparametrii reutilizând aceeași bază de self-play; cere mai multe resurse decât un test rapid.
- **C · 2023 · [Policy-Based Self-Competition for Planning Problems](https://arxiv.org/abs/2306.04403)** — Comparație cu politici istorice; accent pe planificare single-agent, transfer limitat.
- **C · 2020 · [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)** — Referință; NU implementa model de tranziție fiindcă avem simulator exact.
- **C · 2021 · [EfficientZero: Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210)** — Optimizări sample-efficiency pentru modele de dinamică învățată; utilitate redusă aici.
- **C · 2021 · [Learning and Planning in Complex Action Spaces (Sampled MuZero)](https://arxiv.org/abs/2104.06303)** — Acțiuni sample în spațiu mare; noi avem 4.096 indici, dar numai ~12–30 legali.
- **C · 2008 · [Monte Carlo Tree Search: A New Framework for Game AI](https://doi.org/10.1609/aiide.v4i1.18700)** — Fundamente MCTS utile la auditul implementării și al statisticilor.
- **C · 2008 · [Parallel Monte-Carlo Tree Search](https://doi.org/10.1007/978-3-540-87608-3_6)** — Strategii de paralelizare CPU pentru generarea datelor; poate produce overhead la jocuri mici.
- **C · 2008 · [Progressive Strategies for Monte-Carlo Tree Search](https://doi.org/10.1142/S1793005708001094)** — Biasuri și explorare; doar pentru teacher offline.
- **C · 2022 · [Monte Carlo Tree Search: A Review of Recent Modifications and Applications](https://doi.org/10.1007/s10462-022-10228-y)** — Taxonomie amplă pentru alte direcții, nu proiect de implementat integral.
- **C · 2023 · [Beyond Games: A Systematic Review of Neural Monte Carlo Tree Search Applications](https://doi.org/10.1007/s10489-023-05240-w)** — Revizuire pentru coverage, utilitate directă limitată.
- **A · 2023 · [Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract-Datasets_and_Benchmarks.html)** — JAX vectorized simulator poate da 10–100× mai mult throughput vs Python în jocurile lor; transferul cere portarea fidelă a regulilor Colony.

### Date si invatare (24)

- **A · 2023 · [Targeted Search Control in AlphaZero for Effective Policy Improvement (Go-Exploit)](https://arxiv.org/abs/2302.12359)** — Repornește self-play din stări arhivate, aprofundează zone neînvățate fără replay inutil de la start.
- **A · 2026 · [Regret-Guided Search Control for Efficient Learning in AlphaZero (RGSC)](https://arxiv.org/abs/2602.20809)** — Rejoacă stări cu regret mare; se poate implementa variantă simplă fără rețea auxiliară de regret.
- **A · 2025 · [Uncertainty-Guided Exploration for Efficient AlphaZero Training](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a02da3fdfd592d9a5273101c3546611-Abstract-Conference.html)** — Ramifică pe poziții incerte și pe runner-up actions; foarte promițător pentru scene fixed + random warmup.
- **A · 2021 · [Online and Offline Reinforcement Learning by Planning with a Learned Model (Reanalyse)](https://arxiv.org/abs/2104.06294)** — Re-etichetează pozițiile vechi cu teacher îmbunătățit, separând generarea de self-play și etichetare.
- **A · 2021 · [Value Targets in Off-policy AlphaZero: A New Greedy Backup](https://doi.org/10.1007/s00521-021-05928-5)** — Testează bootstrapping/backup target vs rezultat terminal; atenție la supraoptimismul unei evaluări imperfecte.
- **A · 2015 · [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)** — Eșantionează poziții cu erori mari; compară TD-error, policy KL și policy disagreement.
- **A · 2020 · [Improved Reinforcement Learning with Curriculum](https://doi.org/10.1016/j.eswa.2020.113515)** — Învață întâi finaluri / situații decisive și propagă competența înapoi spre debut.
- **A · 2026 · [AlphaZero in Sparsely Rewarded Games: Limits and Auxiliary Supervision](https://arxiv.org/abs/2607.08984)** — Dovedește motivul pentru care outcome-only ratează tactici, iar oracle policy auxiliary ajută.
- **A · 2026 · [Engineering Efficient Self-Play Chess: Search, Replay, and Throughput Under Limited Compute](https://arxiv.org/abs/2609.37447)** — **PREPRINT**; Preprint recent: cost per sample, replay, validare și ablații negative; modelul lor de ~6,32M parametri NU încape direct în ONNX de 5 MiB fp32.
- **A · 2026 · [Move Quality Beyond Arena: Pure Distillation for Small Go Networks](https://doi.org/10.21203/rs.3.rs-9749016/v1)** — **PREPRINT**; Preprint fără peer-review confirmat; etichete teacher deep pentru student; rețeaua autorului este ~7M parametri și NU încape direct în limita noastră de 5 MiB fp32.
- **B · 2024 · [Initial State Diversification for Efficient AlphaZero-style Training](https://doi.org/10.3233/ICG-240255)** — Perturbă pornirile în training fără să abandonezi distribuția efectivă Level 4.
- **B · 2024 · [Accelerate Multi-Agent Reinforcement Learning in Zero-Sum Games with Subgame Curriculum Learning](https://doi.org/10.1609/aaai.v38i10.29011)** — Selectează subgames ca exerciții independente; poate completa Go-Exploit pentru finaluri.
- **B · 2020 · [Prioritized Level Replay](https://arxiv.org/abs/2010.03934)** — Rejoacă situațiile unde modelul este slab, analog cu prioritized states.
- **B · 2023 · [Adversarial Policies Beat Superhuman Go AIs](https://proceedings.mlr.press/v202/wang23g.html)** — Modelele puternice pot avea defecte sistematice; red-team self-play și oracle-based trap mining.
- **B · 2025 · [Can Go AIs Be Adversarially Robust?](https://doi.org/10.1609/aaai.v39i26.34980)** — Exemple de evaluare adversarială, robustețe și prevenirea punctelor slabe; cost mare pentru antrenare completă.
- **B · 2023 · [Policy–Value Alignment and Robustness in Search-based Multi-agent Learning](https://arxiv.org/abs/2301.11857)** — Diagnostichează inconsistențe între policy și value; cere verificare metodologică a transferului.
- **B · 2025 · [Analyzing KataGo: A Comparative Evaluation Against Perfect Play](https://doi.org/10.1007/978-3-031-86585-5_4)** — Test de tip oracle: nivelul aparent mare de joc nu implică apărare perfectă.
- **B · 2024 · [Multiagent Gumbel MuZero: Efficient Planning in Combinatorial Action Spaces](https://doi.org/10.1609/aaai.v38i11.29121)** — Variantă pentru acțiuni compuse; extrage ideea pentru două acțiuni consecutive, nu implementa MuZero.
- **B · 2022 · [AlphaZero-Inspired Game Learning: Faster Training by Using MCTS Only at Test Time](https://arxiv.org/abs/2204.13307)** — Studiază alternative fără search intens în antrenare; posibil baseline TD rapid.
- **B · 2025 · [Active Fine-Tuning of Multi-Task Policies](https://openreview.net/forum?id=hlyBdwHBeC)** — Active-query distillation; inspiră alocarea bugetului teacher pe pozițiile informative, domeniu robotica.
- **B · 2025 · [AlphaZero Neural Scaling and Zipf's Law: a Tale of Board Games and Power Laws](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c15a3033686e7999aecd2740c5a7c4-Abstract-Conference.html)** — Arată că rețelele mai mari pot prioritiza greșit stări frecvente; argument pentru sampling orientat la câștig, nu doar loss.
- **B · 2026 · [Reproducing AlphaZero on Tablut: Self-Play RL for an Asymmetric Board Game](https://arxiv.org/abs/2604.05476)** — **PREPRINT**; Preprint despre roluri asimetrice, uitare catastrofală și replay vs adversari istorici; verifică efectul tie-break Magenta.
- **C · 2024 · [A Survey on Self-play Methods in Reinforcement Learning](https://arxiv.org/abs/2408.01072)** — Review larg al self-play; pentru metodologie după candidatele specifice.
- **B · 2026 · [Scalable Decision-Making for Games of Imperfect Information](https://www.nature.com/articles/s41586-026-11036-y)** — 2026 Nature; sample filtering pe advantage mare și simulator accelerat, dar jocul studiat are informație ascunsă.

### Solvers tactici (17)

- **A · 1993 · [Go-Moku Solved by New Search Techniques](https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-001.pdf)** — Threat-space search și proof-number search; inspirație pentru capture/secvențe forțate în 2 acțiuni.
- **A · 1994 · [Proof-number Search](https://doi.org/10.1016/0004-3702(94)90004-3)** — Demonstrații de câștig/pierdere în subarbori tactici; ajută contra value hallucination.
- **A · 2008 · [Monte-Carlo Tree Search Solver](https://doi.org/10.1007/978-3-540-87608-3_3)** — Combinație de MCTS și rezolvarea pozițiilor certe; recomandat pentru offline teacher.
- **A · 2023 · [Proof Number Based Monte-Carlo Tree Search](https://arxiv.org/abs/2303.09449)** — Combină sampling-ul MCTS cu numere de probă, orientat spre găsirea câștigurilor forțate.
- **A · 2014 · [Bitboard Methods for Games](https://doi.org/10.3233/ICG-2014-37202)** — Accelerare nativă pentru joc 8×8; poate fi mai ieftină decât H200 dacă search e blocaj CPU.
- **A · 1970 · [A New Hashing Method with Application for Game Playing (Zobrist hashing)](https://minds.wisconsin.edu/handle/1793/57624)** — Chei de transpoziție pentru caching și identificarea pozițiilor duplicat.
- **B · 2022 · [Combining Monte-Carlo Tree Search with Proof-Number Search](https://arxiv.org/abs/2206.03965)** — Linie de cercetare înrudită cu PNS–MCTS; compară complexitatea cu implementarea simplă.
- **B · 2003 · [An Effective Two-Level Proof Number Search Algorithm](https://doi.org/10.1016/j.tcs.2002.10.006)** — Optimizare PNS pentru stări cu branching mare; aplicabilă doar dacă solverul exact domină costul.
- **B · 1994 · [Proof-Number Search and Transpositions](https://doi.org/10.3233/ICG-1994-17203)** — Problemele re-folosirii probelor în grafuri cu transpoziții; important pentru corectitudine.
- **B · 2010 · [Dealing with Infinite Loops, Underestimation, Overestimation of DFPN](https://doi.org/10.1609/aaai.v24i1.7534)** — Capcane ale proof-number search; evită atribuire falsă de certitudine.
- **B · 2008 · [About the Completeness of Depth-First Proof-Number Search](https://doi.org/10.1007/978-3-540-87608-3_14)** — Limite de completitudine și cicluri ale DFPN; util numai pentru exact engine robust.
- **B · 2007 · [Checkers Is Solved](https://doi.org/10.1126/science.1144079)** — Metodologie de căutare cu proofs, tabele de finaluri și verificare, nu transfer direct.
- **C · 2007 · [Using Bitboards for Move Generation in Shogi](https://doi.org/10.3233/ICG-2007-30103)** — Idei hardware-specific pentru move generation, folosite numai dacă rule generator e bottleneck.
- **B · 2000 · [Computer Chess Move-ordering Schemes Using Move Influence](https://doi.org/10.1016/S0004-3702(00)00026-6)** — Ordine de explorare a acțiunilor bazată pe amenințări; potrivită pentru tactici forcing.
- **A · 2023 · [Game Solving with Online Fine-Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b663eb1512ce6c268e3e56f34c6d2959-Abstract-Conference.html)** — Proof-cost search în poziții greu de rezolvat și fine-tuning în timpul căutării; teacher offline adaptiv.
- **A · 2023 · [Deep Dive on Checkers Endgame Data](https://webdocs.cs.ualberta.ca/~mmueller/ps/2023/2023_Jiuqi_COG.pdf)** — Învață valori aproape exacte de la tablebase pe eșantioane mici; analogia directă pentru finaluri Colony.
- **A · 2022 · [On the Road to Perfection? Evaluating Leela Chess Zero Against Endgame Tablebases](https://doi.org/10.1007/978-3-031-11488-5_13)** — Măsoară erori ale modelului neural contra tablebase exact; justifică oracle tactical labels.

### Distilare (14)

- **A · 2015 · [Policy Distillation](https://arxiv.org/abs/1511.06295)** — Policy teacher→student, clase legale, soft targets, ablație temperatură/logit margini.
- **A · 2015 · [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)** — Soft targets vs hard labels, temperature și amestec de semnale cu finalul real.
- **A · 2024 · [Amortized Planning with Large-Scale Transformers: A Case Study on Chess (ChessBench)](https://arxiv.org/abs/2402.04494)** — Distilare engine de șah puternic în predictor; extrage Q(s,a)/value labeling, nu scala transformerului.
- **A · 2015 · [Learning to Search Better Than Your Teacher](https://arxiv.org/abs/1502.02206)** — Student poate depăși imitarea prin învățare distribuțională adaptată policy improvement.
- **B · 2022 · [Decoupled Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)** — Separă target/non-target; potențial pentru distribuții rare peste acțiuni legale, de validat.
- **B · 2014 · [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)** — Distilare intermediară dacă rețeaua teacher este și ea neurală; nu funcționează direct cu solver simbolic.
- **B · 2018 · [Born Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html)** — Self-distill student→student; merită doar dacă există timp pentru o iterație suplimentară.
- **B · 2022 · [Knowledge Distillation: A Good Teacher Is Patient and Consistent](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html)** — Coerență augmentări și label-uri; verifică dacă soft-targeturile nu diferă arbitrar.
- **C · 2019 · [Relational Knowledge Distillation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html)** — Păstrează relații între exemple, cost de implementare probabil nejustificat.
- **C · 2018 · [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf)** — Multiple rețele simultan: mai costisitor decât o student compactă.
- **B · 2024 · [Evidence of Learned Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2406.00877)** — Arată că un model poate internaliza tactici/lookahead; argument pentru date cu secvențe forțate.
- **C · 2025 · [Mastering Board Games by External and Internal Planning with Language Models](https://proceedings.mlr.press/v267/schultz25a.html)** — Inspirație pentru Astra/Opus în design și planificare; LLM direct inference prea scump.
- **C · 2025 · [Implicit Search via Discrete Diffusion: A Study on Chess](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f93df618c6907bc0a03222040d70d004-Abstract-Conference.html)** — Direcție exploratorie; probabil incompatibilă cu deadline-ul și ONNX mic.
- **C · 2025 · [PokéChamp: an Expert-level Minimax Language Agent](https://proceedings.mlr.press/v267/karten25a.html)** — Minimax + LLM ca modules; bun ca analogie, dar nu recomand outputuri LLM ca target-uri.

### Arhitectura ONNX (13)

- **A · 2026 · [Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/abs/2605.19091)** — Square tokens + policy attention sursă-destinație; foarte relevant pentru vectorul nostru 64×64, dar un transformer mare nu încape.
- **A · 2024 · [Mastering Chess with a Transformer Model](https://arxiv.org/abs/2409.12272)** — Geometric relative positional biases pentru mișcări; testează tiny attention vs CNN numai la CPU.
- **A · 2026 · [Optimized Kernels in MobileNet Architectures for Computer Chess](https://doi.org/10.1177/13896911261467835)** — Kernel-uri aliniate geometriei pieselor și inferență rapidă; vezi dacă CPU ONNX câștigă.
- **A · 2015 · [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)** — Baseline CNN ResNet 8×8 compact, parametri și FLOP calculați exact.
- **A · 2018 · [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)** — Context global / număr de piese într-un model CNN, overhead minor numai dacă benchmark confirmă.
- **B · 2016 · [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html)** — Atenție la simetriile reale; mai întâi augmentare cu transformări verificate în referee.
- **B · 2018 · [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)** — Depthwise convoluții pot reduce MAC, dar pe 8×8 CPU pot fi mai lente.
- **B · 2019 · [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)** — NAS/eficiență mobilă, compară cost real ONNX și subsetul permis.
- **B · 2020 · [GhostNet: More Features from Cheap Operations](https://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html)** — Rețea compactă posibil, dar overhead pentru graph poate depăși câștigul.
- **C · 2022 · [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545)** — Arhitecturi vision moderne dimensionate pentru imagini mari, relevanță scăzută 8×8.
- **C · 2018 · [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)** — Pruning poate economisi parametri; nu garantează viteză CPU pe ONNX.
- **B · 2022 · [Chess AI: Competing Paradigms for Machine Intelligence](https://pmc.ncbi.nlm.nih.gov/articles/PMC9025083/)** — Comparație critică AlphaZero/Stockfish și reprezentări geometrice ale acțiunilor.
- **A · 2024 · [Enhancing Chess Reinforcement Learning with Graph Representation (GATEAU)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/008a16ead32f932b711788c276890456-Abstract-Conference.html)** — GNN cu edge features pentru acțiuni între pătrate; extrage parametrizarea mutărilor, dar grafurile dinamice pot încălca allow-list-ul ONNX.

### Validare si statistica (8)

- **A · 2023 · [Metamorphic Testing of Chess Engines](https://doi.org/10.1016/j.infsof.2023.107263)** — Teste de invariance/differential correctness între simulatoare și poziții transformate.
- **A · 2025 · [Models and Rating Systems for Head-to-Head Competition](https://doi.org/10.1146/annurev-statistics-040722-061813)** — Estimarea competenței din jocuri puține; folosește meciuri paired Cyan/Magenta și CIs.
- **A · 2001 · [The Glicko-2 Rating System](https://www.glicko.net/glicko/glicko2.pdf)** — Înțelege rating deviation, volatilitate, schimbări de model; nu optimiza doar pentru rating instant.
- **B · 2020 · [Sequential Probability Ratio Tests: Conservative and Robust](https://doi.org/10.1177/0037549720954916)** — Teste secvențiale și early stopping pentru compararea candidaților.
- **B · 2005 · [Tuning Evaluation Functions by Maximizing Concordance](https://doi.org/10.1016/j.tcs.2005.09.047)** — Optimizează o evaluare numerică după ranking-ul mutărilor unui teacher; alternativă la BCE/MSE.
- **C · 2007 · [Computing Elo Ratings of Move Patterns in the Game of Go](https://doi.org/10.3233/ICG-2007-30403)** — Metrici locale ale mutărilor versus scoring global, mai mult diagnostic.
- **C · 2010 · [Whole-History Rating](https://www.remi-coulom.fr/WHR/)** — Referință tehnică de rating longitudinal (nu articol peer reviewed); arena ranking.
- **C · 2008 · [Bayesian Elo Rating](https://www.remi-coulom.fr/Bayesian-Elo/)** — Referință software non-academică pentru perechi de adversari/ci, nu decisivă.

### Referinte tehnice (5)

- **A · 2021-2026 · [KataGo Methods (implementation notes)](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)** — IMPLEMENTARE, nu paper: policy surprise weighting, playout cap, pruning și heuristicile antrenării.
- **A · 2024-2026 · [Leela Chess Zero AlphaZero Primer](https://lczero.org/dev/lc0/search/alphazero/)** — DOCUMENTAȚIE, nu paper: rolul policy/value heads și legal masking.
- **B · 2024-2026 · [Stockfish NNUE Documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)** — DOCUMENTAȚIE, nu paper: valoare eficientă pentru teacher offline, nu incrementality în evaluator ONNX.
- **B · 2019-2026 · [KataGo Source Code](https://github.com/lightvector/KataGo)** — COD, nu paper: referință pentru criterii de eșantionare și experimentare.
- **B · 2023-2026 · [MiniZero Reference Implementation](https://github.com/rlglab/minizero)** — COD, nu paper: punct de pornire pentru benchmark search/self-play.

## 3. Testele / ablațiile decisive (nu înlocui cu presupuneri)

1. **A1 — Teacher:** shallow MCTS vs deep MCTS vs Gumbel vs tactical-augmented deep; măsoară întârziere, întărirea scorului root Q și top-1 acord cu oracle pe poziții de test.
2. **A2 — Date:** uniform self-play vs prioritized replay KL / regret vs go-exploit restart vs endgame curriculum, la același număr de evaluări de nod.
3. **A3 — Label:** terminal z vs search Q vs mix z/Q + exact tactical label; efect pe value calibration, policy top-3 recall, și W/L din arena.
4. **A4 — Policy head:** dense 4.096 vs from–to low rank cu interacțiune QK și bias de poziție relativă, toate pe **set identic de date**.
5. **A5 — Capacitate:** 3,4,6 residual blocks × 32,48,64 canale, plus tiny attention, luând în calcul ONNX file bytes + CPU batch 8 latency.
6. **A6 — Hard cases:** atac Kernel, două acțiuni pentru forțarea sacrificiului, nonalternating perspective, Trace diagonal shoot 2-3, Daemon orthogonal leap range 2-3, Ice path rules, firewall 24→26.
7. **A7 — Evaluation:** 100-simulation head-to-head fixed PUCT, perechi aceleași seed-uri cu schimbarea Cyan/Magenta, CI/elo; nu folosi doar mae/accuracy.
8. **A8 — Perturbare/calibrare:** prior temperature pe logituri, policy entropy target și valoare clipping, eval separate pentru primele 8 acțiuni cu Dirichlet noise și pentru midgame greedy.

## 4. Ce trebuie RESPINS explicit dacă nu există dovezi

- Training direct într-un model de dimensiune LLM sau distilare de text/reasoning în structura ONNX fără game oracle; sursa label-urilor trebuie verificată.
- Schimbarea MCTS-ului referee sau ajustarea PUCT în fișierul ONNX: imposibil prin contractul acceptat.
- Reimplementarea MuZero pentru a învăța tranzițiile jocului: redundanță față de simularea exactă disponibilă.
- H200 doar pentru „mai mult GPU”: când bottleneck-ul este move generation/search pe CPU, costul crește fără date mai multe.
- Arhitecturi sofisticate fără inferență mai rapidă și fără diferență semnificativă în paired tournament.
- Rotiri/flipuri arbitrare fără verificare că regulile și side-to-move plane păstrează identic valoarea și legal moves.

## 5. Prompt unificat pentru Astra și Opus — aceeași cerință, evaluări independente

```text
Ești cercetător independent în reinforcement learning, game-tree search și optimizare neural inference.
Obiectiv: proiectează cea mai puternică soluție ONNX care poate câștiga Colony: One Level 4.
Nu optimiza pentru eleganță teoretică sau publicație. Optimizează win rate, reliability, time-to-ship.

DOCUMENTE obligatorii: acest dosar bibliografic + toate cele 4 ZIP-uri din concurs;
citește în special rules.py, env.py, mcts.py, encoding.py, docs/submission.md, level-4_scene.json.
Nu extrapola din șah/Go fără ipoteză falsificabilă și evaluare.

SARCINI:
1. Auditează complet regulile: cele 6 tipuri de piese, două acțiuni/tură, protectori Kernel,
   shrinks/tie-break, perspectiva scorului, mapping 4096 policy, constrângerile CPU/ONNX.
2. Din bibliografia de mai sus identifică TOP 15 mecanisme cu relevanță concretă;
   separă tactic solver, training data, policy head, value head, search oracle, runtime.
3. Pentru fiecare: mecanism matematic, presupuneri, cost (ore CPU/GPU),
   beneficiu probabil, risc de implementare, mod de verificare, ce ablație decide adoptarea.
4. Propune o arhitectură de referință care încape sub 5 MiB și trece exact validatorul.
   Compară ResNet-conv cu origine/destinație attention bilinear și cu un head dense.
5. Definește un teacher exact sau aproximativ credibil, prioritate pentru tactici de
   două acțiuni, amenințări imediate, firewalls, terminal proven, bun raport nod/secundă.
6. Definește dataset collector, curriculum, prioritized re-labeling, policy targets legal-only,
   value targets side-to-move cu semn schimbat numai la schimbare de tură.
7. Proiectează un tournament arena de paired seeds Cyan/Magenta, 100 simulations,
   PUCT/noise identice cu referee, metrici win rate + CI și model-selection robust.
8. Livrabile: RANKING PROPOSALS / SYSTEM DESIGN / MODEL ARCH / DATA PIPELINE /
   ABLATIONS / FAILURES / TIME-BUDGET PLAN / ANNOTATED SOURCE REFERENCES.
9. Critică activ fiecare idee scumpă fără evidence (H200, MuZero, giant transformers).
10. Nu face false claims de peer-review, speedup sau Elo; marchează preprinturile.
```

## 6. Reguli de evaluare a bibliografiei

- **Tier A:** include o idee implementabilă sau o avertizare deosebit de importantă pentru jocul exact (nu este garanție de câștig).
- **Tier B:** alternativă / ablație după baseline; trebuie justificată prin calcul și experiment.
- **Tier C:** fundal academic, comparator sau idee dificil de transferat; nu copia direct.
- **Tip sursă:** „Referințe tehnice” conține explicit documentație și cod, nu articole peer reviewed; dintre celelalte există preprinturi nevalidate editorial.
- **Definiția succesului:** câștig în paired head-to-head sub MCTS-ul referee și validare ONNX/CPU; bibliografia nu poate garanta un câștig.

## 7. Atenție la citare și verificare

Linkurile trimit la pagini de arXiv, pagini oficiale ale conferințelor/jurnalelor, DOI sau documentație originală. Lucrarea Gumbel AlphaZero: linkul original verificat este https://openreview.net/pdf?id=bERaNdoegnO (nu arXiv 2202.11772). S-au verificat prin căutare și consultarea surselor principale o selecție amplă de lucrări, dar nu s-a verificat integral fiecare PDF. Nu presupune că toate ipotezele fiecărei lucrări transferă în Colony: One; verifică secțiunile Method/Ablations/Limitations înainte de implementare. Pentru lucrări disponibile doar ca preprint, concluziile sunt preliminare.

