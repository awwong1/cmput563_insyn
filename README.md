# INSYN

Github repository containing relevant source code for University of Alberta, CMPUT 563 (Probabilistic Graphical Models) project material.

## Requirements

Ensure all requirements are met before quickstart.

- [`python3`](https://www.python.org/)
- [`pipenv`](https://github.com/pypa/pipenv)
- [`antlr`](https://www.antlr.org/)
- [`sqlite3`](https://www.sqlite.org/index.html)

```bash
$ python3 --version
Python 3.6.7

$ pipenv --version
pipenv, version 11.9.0

$ antlr4
ANTLR Parser Generator  Version 4.7.1

$ sqlite3 -version
3.25.3 2018-11-05 20:37:38 89e099fbe5e13c33e683bef07361231ca525b88f7907be7092058007b750alt1
```

## Quickstart

1. Ensure the required sqlite files are available in `./data` (see [`./data/README.md`](data/README.md))
2. Initialize and start pip environment.
    - `pipenv install`
    - `pipenv shell`
3. Run `./insyn.py`
4. If using 10-gram model, follow instructions in `./model` (see [`./model/README.md`](model/README.md))

```bash
$ ./insyn.py 
usage: insyn.py [-h] [-l level] [--test-ngram-model file|dir]
                [--sample-parse [offset]] [--generate-structure]
                [--tokenize-training-data]

Reccomendation Models for Syntactically Incorrect Source Code

optional arguments:
  -h, --help            show this help message and exit
  -l level, --log level
                        set logging verbosity
  --test-ngram-model file|dir
                        read java code, change random token, list suggestions
  --sample-parse [offset]
                        sample output sequence from training db
  --generate-structure  generate HHMM structure from grammar
  --tokenize-training-data
                        tokenize all training data
```

Example ngram validation on one file (also handles directories, does recursive walk for all `*.java` files)

```bash
$ ./insyn.py --log info --test-ngram-model example/HelloWorld.java
INFO:analyze.ngram_tester:example/HelloWorld.java: BREAK by ADD LT at 22
INFO:analyze.ngram_tester:example/HelloWorld.java: CHECKING_LOCATION 17 (-3.151554584503174)
INFO:analyze.ngram_tester:example/HelloWorld.java: CHECKING_LOCATION 26 (-2.4582955837249756)
INFO:analyze.ngram_tester:example/HelloWorld.java: CHECKING_LOCATION 0 (-2.2994213104248047)
INFO:analyze.ngram_tester:example/HelloWorld.java: CHECKING_LOCATION 24 (-1.8517078161239624)
INFO:analyze.ngram_tester:example/HelloWorld.java: CHECKING_LOCATION 22 (-1.6897526979446411)
INFO:analyze.ngram_tester:example/HelloWorld.java: SUGGEST_FIX_SCORE -12.189971923828125 (DEL LT at 22)
INFO:analyze.ngram_tester:example/HelloWorld.java: TRUE_FIX_FOUND rank: 0
example/HelloWorld.java: ADD LT at 22; True fix found rank 0
Found 1/1 true fixes (avg_rank=0.0)
```

When performing evaluation, ensure that the code has not been used for training!
```sql
sqlite> PRAGMA table_info(repository_source);
0|owner|VARCHAR|1||1
1|name|VARCHAR|1||2
2|hash|VARCHAR|1||3
3|path|VARCHAR|1||4

SELECT COUNT(*) FROM repository_source WHERE owner = 'TheAlgorithms';
41
```

[MIT License](LICENSE).
