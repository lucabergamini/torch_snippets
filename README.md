# torch_snippets

Just some Pytorch snippets.

## PRO
- Non è interpretato come gli altri, quindi in forward si vede sempre tutto
- Si collega a tensorboard
- torch mappa direttamente gli array numpy quindi si puo fare tutto senza problemi
- non bisogna usare cicli e condizioni come funzioni (scan, ifelse,ecc..)

## CONTRO
- Non sa le dimensioni dei tensori, nei layer connessi vanno scritte a mano (nel primo almeno)
- potrebbe essere lento
- alcuni problemi di visualizzazione del grafo in tensorboard
- non ha il logger, va scritto con progressbar (già fatto)
- ~~ha un sistema di scruttura dei layer poco scalabile~~

## BEST PRACTICES
- usa programmazione ad oggetti per blocchi
- se un blocco è sequenziale eredita da Sequential
- quando tutto funziona usa inplace su ReLU per diminuire uso memoria
- ModuleList equivale ad aggiungere moduli e chiamarli con children
- Visto che ti servono le dimensioni calcolale nell init e falle fluire

``` self.layer1 =  nn.Sequential(*list(model.children())[:k[0]]) ```
questo permette di avere accesso all intermedio, quindi devo fare subclass
