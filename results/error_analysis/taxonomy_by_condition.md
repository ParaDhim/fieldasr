Primary tag among failures (item WER >= 0.5), whisper_lora_s0

| condition        |   hallucination_loop |   noise_or_channel |   numeral |   orthographic_near_miss |   proper_noun |   speaker_or_unclear |   n_failures |   failure_rate% |
|:-----------------|---------------------:|-------------------:|----------:|-------------------------:|--------------:|---------------------:|-------------:|----------------:|
| amr_nb_8k        |                    9 |                 50 |        12 |                       10 |            58 |                    0 |          139 |            28.5 |
| babble_snr0      |                   81 |                147 |        49 |                        1 |           136 |                    0 |          414 |            85   |
| babble_snr10     |                    4 |                 52 |        15 |                       10 |            61 |                    0 |          142 |            29.2 |
| cheap_mic_reverb |                    3 |                 96 |        27 |                       17 |           102 |                    0 |          245 |            50.3 |
| clean            |                    0 |                  0 |         2 |                        5 |            15 |                    7 |           29 |             6   |
| domain_terms     |                    0 |                  1 |         2 |                        1 |             2 |                    0 |            6 |             5   |
| mp3_8k           |                    0 |                 17 |         8 |                        9 |            24 |                    0 |           58 |            11.9 |
| pink_snr0        |                   37 |                105 |        39 |                        7 |           138 |                    0 |          326 |            66.9 |
| pink_snr10       |                    5 |                 47 |        12 |                       11 |            61 |                    0 |          136 |            27.9 |

Overall (primary tag): {'proper_noun': 597, 'noise_or_channel': 515, 'numeral': 166, 'hallucination_loop': 139, 'orthographic_near_miss': 71, 'speaker_or_unclear': 7}

Non-exclusive prevalence (an item can carry several tags; 'proper_noun' fires whenever any capitalised mid-sentence word is missed, so it is inflated on long sentences):

| tags                   |   n_failures |
|:-----------------------|-------------:|
| noise_or_channel       |         1466 |
| proper_noun            |          784 |
| numeral                |          197 |
| hallucination_loop     |          139 |
| orthographic_near_miss |          124 |
| foreign_token          |           30 |
| speaker_or_unclear     |            7 |
