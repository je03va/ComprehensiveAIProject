# Findings: can off-the-shelf sentiment models understand developers?

Sample: 1000 rows per dataset. Models: textblob, vader, vader_tuned, roberta.

## Best model per dataset (macro F1)
- **github**: best = roberta (F1 0.76, accuracy 0.76); worst = vader (F1 0.65).
- **stackoverflow**: best = roberta (F1 0.78, accuracy 0.78); worst = textblob (F1 0.69).

## Where general-purpose models fail on developer language
- **github**: VADER calls 23% of truly neutral posts negative; TextBlob 8%; RoBERTa 26%.
- **stackoverflow**: VADER calls 21% of truly neutral posts negative; TextBlob 11%; RoBERTa 18%.
- On 8 hand-picked dev sentences, plain VADER labels 6 as negative; RoBERTa labels 4 as negative. Example: "Kill the process and restart the server." -> VADER -0.69, RoBERTa 'negative'.

## Tech-lexicon tuning (VADER before vs after)
- **github**: macro F1 0.65 -> 0.68 (+0.04); neutral posts called negative 23% -> 18%.
- **stackoverflow**: macro F1 0.71 -> 0.73 (+0.01); neutral posts called negative 21% -> 20%.
- On the example sentences, tuning changed 6 of 8 labels (see outputs/tuning_examples.csv).

## Model agreement (TextBlob, VADER, RoBERTa)
- **stackoverflow**: all 3 agree on 57% of posts; when they agree they are right 91% of the time. Examples of disagreement: outputs/disagreements_stackoverflow.csv.
- **github**: all 3 agree on 48% of posts; when they agree they are right 94% of the time. Examples of disagreement: outputs/disagreements_github.csv.

## Runtime (seconds per dataset sample)
- roberta: github 68.3s, stackoverflow 106.9s
- textblob: github 0.2s, stackoverflow 0.3s
- vader: github 0.1s, stackoverflow 0.1s
- vader_tuned: github 0.1s, stackoverflow 0.1s
