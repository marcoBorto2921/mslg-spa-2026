Domande — Sezione A: task e metriche ufficiali

     1. Qual e' la metrica ufficiale del ranking IberLEF 2026 MSLG-SPA? Z-score su BLEU+METEOR+chrF+COMET o c'e' una combinazione diversa?
     2. Per la subtask SPA2MSLG, COMET e' escluso? Confermarlo dal testo del task.
     3. Qual e' il formato esatto atteso del submission file (estensione, encoding, quote, newline, naming convention)?
     4. Quante submission per team sono ammesse e qual e' il deadline esatto (data + timezone)?
     5. Il test set ha reference nascoste o e' blind? Ci sono dev/devtest set intermedi?
     6. Il task ammette uso di LLM closed-source (GPT-4, Claude) o solo modelli open?
     7. Ci sono restrizioni sull'uso di dati esterni (monolingual Spanish, altri sign language corpora)?

     Domande — Sezione B: dataset e annotazioni MSL

     8. Come sono stati raccolti i 490 pairs? (traduttori umani, video annotati, crowdsourcing?)
     9. Qual e' la semantica esatta delle annotazioni speciali: dm-, +, #, -, maiuscole? (es. dm- = dominant hand? + = compound? # = fingerspelling?)
     10. Esistono split ufficiali train/dev/test o e' tutto "train" e il test arrivera' separato?
     11. Qual e' la distribuzione dei domini (dialoghi, frasi, racconti)?
     12. Esistono linee guida di annotazione pubbliche che possono guidare il preprocessing?
     13. Il Mexican Sign Language (LSM) condivide annotazioni con ASL, LSE (Spagnolo), altri? Potrei fare pretraining intermedio?

     Domande — Sezione C: tecniche per low-resource MT

     14. Qual e' lo stato dell'arte per traduzione low-resource (<1000 pairs)? Ci sono paper specifici su sign language gloss translation?
     15. Quali sono i risultati migliori ottenuti su RWTH-PHOENIX-2014T (gloss→tedesco) e con quali tecniche? E' comparabile a MSLG→SPA?
     16. Quali target_modules LoRA funzionano meglio per mBART in traduzione? Solo q,v o anche k,o,fc1,fc2?
     17. Quale range di LoRA rank e' ottimale per dataset <500 pairs? r=8, 16, 32, 64?
     18. Back-translation: quante coppie sintetiche sono "troppo" rispetto al dataset reale? Quale rapporto sintetico/reale massimizza le
     prestazioni?
     19. Iterative back-translation (Hoang et al. 2018): vale la pena con questa taglia di dataset?
     20. Esistono tecniche di data augmentation specifiche per glosse (word dropout, subword swap, synonym replacement)?
     21. Curriculum learning (short→long, o easy→hard) funziona in low-resource MT?
     22. Label smoothing e' stato mostrato efficace su mBART fine-tuning?

     Domande — Sezione D: ensembling e decoding

     23. Oltre a checkpoint ensemble top-k, quali strategie di ensembling hanno funzionato in shared task IberLEF/WMT low-resource? (MBR decoding,
     n-best rerank con COMET, seed ensemble)
     24. Minimum Bayes Risk decoding con chrF come utility function e' implementabile con sacrebleu? Delta atteso?
     25. Length penalty e no_repeat_ngram_size consigliati per mBART su sequenze corte (<15 token)?
     26. Beam size ottimale: 5 e' sufficiente o conviene 10-20 per sequenze corte?

     Domande — Sezione E: paper submission

     27. IberLEF richiede un working-notes paper? Qual e' il template, la lunghezza massima, il deadline?
     28. Quali sezioni sono obbligatorie (system description, results, error analysis)?
     29. Ci sono esempi di working notes dell'edizione 2025 dello stesso task o task simili che posso consultare per calibrare il livello?
     30. Error analysis tipica per low-resource MT: quali categorie di errore sono utili da riportare (hallucination, omission, lexical choice,
     agreement)?

     Domande — Sezione F: risk check finale

     31. Ci sono stati cambiamenti recenti alle regole del task (rispetto all'annuncio iniziale)?
     32. Qualche team ha pubblicato risultati preliminari o sistemi baseline ufficiali che fissano una soglia minima?
     33. La metrica COMET usa quale modello (wmt22-comet-da, xcomet, Unbabel/XCOMET-XL)? Serve licenza?

     ---
     Parte 3 — Workflow suggerito dopo NotebookLM

     1. Rispondere alle domande A (metriche/format) → bloccante, se sbaglio il format perdo tutto.
     2. Rispondere alle domande B (annotazioni MSL) → guida l'introduzione di special tokens piu' efficaci del solo [HYPHEN].
     3. Rispondere alle domande C (tecniche low-resource) → guida il prossimo ciclo di esperimenti (LoRA target_modules, rank, label smoothing,
     iterative BT).
     4. Eseguire EXP-001 in parallelo (non dipende da NotebookLM).
     5. Migliorare SPA2MSLG (dipende da C16, C17).
     6. Rigenerare back-translation (dipende da C18, C19).
     7. Scrivere working notes (dipende da E27-30).



     RISPOSTE DI NOTEBOOOK LM

     Sezione A :
     In base ai documenti forniti, ecco le risposte ai punti della tua richiesta relativi al task **IberLEF 2026 MSLG-SPA** (Bidirectional Translation between Mexican Sign Language Glosses and Spanish):

1.  **Metrica ufficiale del ranking:** Le fonti indicano che la valutazione viene condotta tipicamente utilizzando l'**accuratezza** per i subtask e le metriche **BLEU** e **TER** per l'output di traduzione principale. Viene inoltre suggerito l'uso di **chrF** a livello di n-grammi di caratteri, poiché più robusto rispetto alle variazioni morfologiche dello spagnolo e ai marcatori simbolici della LSM. **Non viene fatta menzione** di una combinazione basata su Z-score (BLEU+METEOR+chrF+COMET) né del parametro METEOR nei testi analizzati.
2.  **Esclusione di COMET per SPA2MSLG:** Le fonti discutono BLEU, TER e chrF come metriche di valutazione per il task, ma **non citano mai COMET** nel contesto della traduzione verso le glosse (SPA2MSLG) o in generale per la sfida MSLG-SPA. L'assenza di riferimenti a COMET nei documenti tecnici suggerisce che non sia inclusa o considerata standard per questo specifico scenario di traduzione glossa-testo.
3.  **Formato esatto del submission file:** Le fonti non specificano il formato esatto (estensione, encoding, naming convention, ecc.) atteso per i file di sottomissione. Vengono citati in generale file CSV per set di dati di riconoscimento, ma non sono fornite le linee guida tecniche per l'invio degli output del sistema al challenge IberLEF 2026.
4.  **Numero di submission e deadline:**
    *   **Numero di submission:** Il limite di sottomissioni per team non è riportato nelle fonti.
    *   **Deadline:** La scadenza per la consegna degli output e dei "run" del sistema è fissata per il **30 aprile 2026**.
    *   **Timezone:** I documenti non specificano il fuso orario esatto (sebbene il workshop IberLEF si tenga in Spagna).
5.  **Test set e set intermedi:** La fase di test inizia il **27 marzo 2026** con il rilascio del test set. Le fonti indicano che i dati di addestramento (Training data) vengono resi disponibili il **12 marzo 2026**, ma non specificano l'esistenza di set di sviluppo (dev/devtest) intermedi rilasciati ufficialmente dagli organizzatori del task, pur essendo prassi comune nella ricerca citata utilizzare partizioni interne per la validazione.
6.  **Uso di LLM closed-source:** Non ci sono informazioni nelle fonti che proibiscano o autorizzino esplicitamente l'uso di modelli a codice chiuso come GPT-4 o Claude. Tuttavia, la letteratura tecnica fornita mostra l'uso frequente di modelli open-source come **mBART** e **Llama 3**. Un documento menziona l'uso di GPT-4 specificamente per la generazione di parafrasi sintetiche in una strategia di data augmentation, ma non come vincolo o regola del task IberLEF.
7.  **Restrizioni sull'uso di dati esterni:** Le fonti non elencano restrizioni ufficiali; al contrario, evidenziano come l'uso di **corpora esterni**, come quello della lingua dei segni americana (ASL, es. ASLG-PC12) o dati monolingua in spagnolo, sia una strategia caldamente consigliata e potenziante per affrontare la scarsità di risorse della LSM. In particolare, l'integrazione di dati ASL ha dimostrato incrementi significativi nel punteggio BLEU (da 62 a 85) in esperimenti di traduzione della lingua dei segni messicana.

Sezione B : 
Ecco le informazioni relative al dataset e alle annotazioni per il task **IberLEF 2026 MSLG-SPA**, basate sui documenti forniti:

8.  **Raccolta dei dati:** Sebbene la tua richiesta citi "490 coppie", le fonti descrivono principalmente un corpus di **3.000 coppie di frasi allineate** (Spagnolo - glosse LSM) costruito da Lara-Ortiz et al. (2025). Questi dati sono stati raccolti con il supporto e la partecipazione attiva del **"Grupo Promotor de la LSM"**, composto da persone sorde messicane e interpreti di lingua dei segni, garantendo l'allineamento con le prospettive della comunità.
9.  **Semantica delle annotazioni speciali:** I simboli utilizzati hanno funzioni linguistiche precise per catturare le sfumature della LSM:
    *   **`dm-` (Discourse / Deictic):** Contrassegna i **dimostrativi** (es. *dem:1*, *dem:2*) o il puntamento a livello di discorso per il tracciamento dei referenti nello spazio.
    *   **`+` (Compound / Reduplication):** Indica un **concetto composto** formato da due segni o un movimento ripetuto per esprimere un significato aspettuale.
    *   **`#` (Lexicalized Loan):** Denota parole dello spagnolo prese in prestito tramite l'alfabeto manuale (*fingerspelling*) che sono state ristrutturate fonologicamente in un **unico segno** (es. *#VW* per Volkswagen).
    *   **`-` (Multi-word / Affix):** Utilizzato per collegare più parole spagnole che corrispondono a un solo segno o per identificare **attacchi morfologici**.
    *   **Maiuscole:** Vengono utilizzate per le **ID-gloss**, ovvero etichette testuali che approssimano il significato dei segni senza l'interferenza della grammatica della lingua parlata.
10. **Split ufficiali:** Secondo il calendario del task, i dati di addestramento (**Training data**) vengono rilasciati il **12 marzo 2026**, mentre il set di test (**Test set**) viene reso disponibile separatamente il **27 marzo 2026**. Negli studi preliminari su questo corpus, i ricercatori hanno utilizzato partizioni interne (80% training, 10% validation, 10% test) per la validazione.
11. **Distribuzione dei domini:** Il lato LSM è caratterizzato da sequenze di glosse **molto brevi e compresse**, con una mediana di circa **3 token per frase** (rispetto ai 4 token del lato spagnolo). Gli esempi forniti includono frasi di uso comune e descrizioni di stati (es. *"Ellas viven en México"*, *"La niña está loca"*). Non viene fornita una percentuale esatta tra dialoghi e racconti, ma la brevità suggerisce una prevalenza di frasi isolate o scambi diretti.
12. **Linee guida di annotazione:** Il documento tecnico *"Neural Architectures and Linguistic Paradigms..."* discute le logiche simboliche e le linee guida seguite per la costruzione del corpus di Lara-Ortiz et al., che funge da riferimento per il challenge.
13. **Condivisione di annotazioni e pretraining:** La LSM e la lingua dei segni americana (ASL) sono entrambe derivate dalla lingua dei segni francese (**LSF**) e condividono somiglianze strutturali, come l'ordine dei componenti **SOV** (Soggetto-Oggetto-Verbo). Gli esperimenti dimostrano che il **pretraining intermedio** (o l'aumento dei dati) tramite il corpus ASLG-PC12 (English-ASL) è estremamente efficace: l'integrazione di dati ASL ha permesso di elevare il punteggio BLEU della traduzione LSM da 62 a **85**.


 RISPOSTESEZIONE C: 
Ecco un'analisi dello stato dell'arte e delle tecniche avanzate per la traduzione automatica in scenari a bassissime risorse, con particolare attenzione alle glosse della lingua dei segni:

14. **Stato dell'arte per <1000 coppie e Gloss Translation:** Lo stato dell'arte per dataset estremamente ridotti è passato dall'addestramento da zero (from-scratch) al **fine-tuning di modelli linguistici pre-addestrati (PLM)** come **T5, mBART e Llama**. Ad esempio, sul dataset **SIGNUM** (circa 600 coppie), il modello mBART raggiunge punteggi di **67.60 BLEU-4** nella traduzione da glossa a testo. Esistono paper specifici che esplorano la traduzione bidirezionale delle glosse utilizzando PLM, evidenziando come questi modelli possano trasferire la conoscenza linguistica pre-acquisita per mappare glosse su frasi naturali.

15. **Risultati su RWTH-PHOENIX-2014T:** I risultati migliori per la traduzione glossa→tedesco su questo dataset vedono **Llama 8B** raggiungere **29.92 BLEU-4** e **mBART** arrivare a **25.58**. Le tecniche vincenti includono l'uso di modelli con architettura encoder-decoder e l'integrazione di obiettivi di denoising. Rispetto a **MSLG→SPA**, PHOENIX-14T è comparabile per la natura del dominio (meteo) e la divergenza sintattica, ma il dataset messicano è circa la metà in termini di dimensioni (3.000 coppie vs circa 7.000) e presenta glosse molto più brevi (mediana di 3 token).

16. **Target_modules LoRA per mBART:** Sebbene la letteratura classica si concentri spesso su `q` e `v`, gli esperimenti più recenti di adattamento efficiente (come nel framework **LowRA**) suggeriscono che l'applicazione a **tutti i moduli lineari** (`all_linear` o moduli `q, k, v, o, fc1, fc2`) garantisca una migliore stabilità e performance in scenari a basse risorse.

17. **LoRA Rank ottimale per <500 coppie:** Per dataset molto piccoli, i framework di ottimizzazione suggeriscono l'uso di un **r=64** (con alpha=64) per garantire che il modello abbia sufficiente capacità di assorbire la mappatura glossa-testo senza dover aggiornare l'intero set di parametri, riducendo al contempo il rischio di overfitting tipico del full fine-tuning su dati scarsi.

18. **Rapporto Back-translation sintetico/reale:** Gli esperimenti specifici sulla lingua dei segni messicana hanno dimostrato che l'**oversampling dei dati esterni o sintetici di un fattore 4** (portando il dataset da 3k a circa 12k coppie totali) massimizza le prestazioni, elevando il BLEU da 62 a **85**. Oltre questo rapporto si osservano rendimenti decrescenti.

19. **Iterative back-translation (Hoang et al. 2018):** Questa tecnica è considerata **particolarmente valida** per dataset con meno di un milione di righe. È stata applicata con successo in traduzioni Luganda-Inglese per mitigare la scarsità di dati bi-testuali, suggerendo che valga la pena implementarla anche per taglie ridotte come quella del task MSLG-SPA.

20. **Data augmentation specifica per glosse:** Oltre alla back-translation, le tecniche includono:
    *   **Parafrasi tramite LLM:** Uso di GPT-4 per generare varianti della lingua target preservando il significato.
    *   **Strategie basate su lemmi:** Approssimazione delle glosse tramite lemmatizzazione e rimozione di articoli e ausiliari.
    *   **Sostituzione di sinonimi e swapping:** Tecniche di alterazione testuale per aumentare la varietà lessicale.

21. **Curriculum learning:** Questa tecnica viene citata come efficace per l'adattamento di dominio nella NMT, suggerendo che un approccio di apprendimento progressivo (da frasi semplici a complesse) possa aiutare in scenari low-resource.

22. **Label smoothing su mBART:** È stato dimostrato efficace, in particolare attraverso una variante chiamata **Semantically Aware Label Smoothing (SALS)**, progettata specificamente per gestire l'ambiguità intrinseca delle glosse durante il fine-tuning di modelli pre-addestrati.


Sezione D:  risposte
In base ai documenti forniti, ecco le informazioni relative alle strategie di ensembling e alle tecniche di decoding per il task MSLG-SPA e contesti simili a basse risorse:

**23. Strategie di ensembling in IberLEF/WMT low-resource**
Le fonti indicano che, oltre al classico ensembling di checkpoint, una strategia efficace in workshop correlati (come quelli del 2025) è stata l'uso di **architetture ensemble che combinano modelli basati su Transformer con euristiche basate su regole**. Questo approccio è utile per gestire marcatori linguistici specifici della lingua dei segni come `dm-` e `#`.
Inoltre, viene evidenziata l'importanza dell'ensembling per **stabilizzare le prestazioni** dei modelli in scenari a basse risorse. Altre strategie menzionate nella letteratura generale per NMT low-resource includono la **data diversification**, che prevede l'addestramento di molteplici modelli su diverse versioni dei dati per migliorare la robustezza. Non viene invece fatta menzione specifica nelle fonti di "MBR decoding" o "n-best rerank con COMET" applicati a questo specifico task.

**24. Minimum Bayes Risk decoding con chrF e sacrebleu**
I documenti **non contengono informazioni** circa l'implementazione del Minimum Bayes Risk (MBR) decoding tramite `sacrebleu`, né forniscono dati sul delta di miglioramento atteso utilizzando chrF come funzione di utilità. Le fonti confermano l'uso di `sacrebleu` principalmente come strumento standard per il **calcolo delle metriche** di valutazione (BLEU, chrF, TER) alla fine del processo.

**25. Length penalty e no_repeat_ngram_size per mBART su sequenze corte**
Le fonti analizzate **non forniscono raccomandazioni specifiche** sui valori di `length_penalty` o `no_repeat_ngram_size` per il fine-tuning di mBART su sequenze inferiori ai 15 token. Sebbene mBART sia ampiamente discusso per la sua efficacia nel denoising e nella traduzione di glosse, i dettagli tecnici sui parametri di generazione per frasi brevi non sono dettagliati nei testi forniti.

**26. Beam size ottimale per sequenze corte**
I documenti confermano l'uso standard della **beam search** per la generazione di dati sintetici e traduzioni, ma **non specificano una dimensione ottimale del raggio** (beam size) tra 5 e 20 per sequenze corte. Viene solo notato che la beam search è il metodo tipicamente utilizzato rispetto al campionamento casuale, che tende a funzionare meglio solo in scenari ad alte risorse.

In sintesi, i documenti si concentrano maggiormente sulle architetture di addestramento (mBART, LoRA) e sulle strategie di aumento dei dati (back-translation, ASL transfer), lasciando i dettagli specifici del decoding fine-grained alla libera implementazione dei partecipanti.


Ecco le risposte dettagliate riguardanti la sottomissione dei paper e l'analisi finale dei rischi per il task IberLEF 2026 MSLG-SPA:

### rispote — Sezione E: paper submission

27. **Working-notes e scadenze:** IberLEF richiede la sottomissione di **system description papers** (che fungono da working notes) che verranno pubblicati nei relativi atti su **CEUR-WS.org**. La scadenza per la consegna degli output del sistema è il **30 aprile 2026**, mentre la versione finale (camera-ready) del paper deve essere inviata entro il **3 luglio 2026**. I testi non specificano un template unico o una lunghezza massima, ma indicano che gli organizzatori di ogni task gestiscono la valutazione scientifica dei lavori inviati dai partecipanti.
28. **Sezioni obbligatorie:** Sebbene non venga fornito un elenco rigido, i documenti indicano come fondamentale la **descrizione del sistema** (system description). In letteratura correlata, papers accettati in workshop simili (WSLP 2025) includono tipicamente sezioni su: dataset utilizzato, architettura del modello, configurazione sperimentale, **risultati quantitativi** e una discussione sull'efficacia delle tecniche di aumento dei dati.
29. **Esempi di riferimento:** Puoi consultare i lavori accettati al workshop **WSLP 2025** (strettamente correlato), come *"Finetuning Pre-trained Language Models for Bidirectional Sign Language Gloss to Text Translation"* e il paper di **Lara-Ortiz et al.** riguardante proprio la LSM e l'aumento dei dati tramite ASL. Questi esempi mostrano un focus elevato sull'uso di modelli pre-addestrati e tecniche di cross-lingual transfer.
30. **Categorie di errore per low-resource MT:** Per sistemi di traduzione in scenari a scarse risorse come la LSM, è utile riportare errori relativi a:
    *   **Allucinazioni (Hallucinations):** Generazione di testo non presente nella fonte.
    *   **Errori sui nomi propri:** Traduzione errata di entità specifiche.
    *   **Discrepanze grammaticali:** Errori di accordo soggetto-verbo.
    *   **Divergenza semantica totale:** Casi in cui l'output non ha alcuna relazione con il significato originale.
    *   **Difficoltà con numeri e riferimenti tecnici:** Un problema comune rilevato nei sistemi di traduzione delle glosse.

###ì RIPOSTE — Sezione F: risk check finale

31. **Cambiamenti alle regole:** I documenti analizzati non segnalano cambiamenti recenti rispetto agli obiettivi iniziali; il calendario rimane fissato con il rilascio dei dati di addestramento a marzo 2026 e la chiusura della fase di test ad aprile 2026.
32. **Risultati preliminari e baseline:** Esistono già soglie di riferimento basate sul corpus di Lara-Ortiz et al. (2025), che è la base del task:
    *   **Baseline standard (BARTO fine-tuned):** Ha ottenuto un punteggio **BLEU-4 di 35.0**.
    *   **Baseline basata su lemmi (TreeTagger):** Supera il modello neurale base nella precisione dei singoli unigrammi (BLEU-1) ma fallisce sulle sequenze lunghe.
    *   **Soglia potenziata:** Tramite l'integrazione di dati ASL, i ricercatori hanno dimostrato di poter elevare il punteggio **BLEU da 62 a 85**.
33. **Specifica COMET:** Le fonti fornite **non contengono informazioni** sul modello specifico di COMET (es. wmt22-comet-da o xcomet) né sui requisiti di licenza, in quanto la metrica non è citata nei documenti tecnici relativi alla valutazione ufficiale di questo specifico task.