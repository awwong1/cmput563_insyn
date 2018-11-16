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

```bash
$ ./insyn.py 
usage: insyn.py [-h] [-v] [-p] [-o DB_OFFSET]

Reccomendation Models for Syntactically Incorrect Source Code

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase output verbosity
  -p, --parse-example   example output sequence from sqlite3 db
  -o DB_OFFSET, --db-offset DB_OFFSET
                        in example, number of sqlite3 db rows to offset
```

```bash
$ ./insyn.py --parse-example --db-offset 0
============== SOURCE CODE ==============
package com.cl.interpolatordebugger;

import android.app.Application;
import android.test.ApplicationTestCase;

/**
 * <a href="http://d.android.com/tools/testing/testing_android.html">Testing Fundamentals</a>
 */
public class ApplicationTest extends ApplicationTestCase<Application> {
    public ApplicationTest() {
        super(Application.class);
    }
}

============== PARSED CODE ==============
{<R:0> {<R:1> <T:32> {<R:46> <T:111> <T:69> <T:111> <T:69> <T:111> } <T:67> } {<R:2> <T:25> {<R:46> <T:111> <T:69> <T:111> <T:69> <T:111> } <T:67> } {<R:2> <T:25> {<R:46> <T:111> <T:69> <T:111> <T:69> <T:111> } <T:67> } {<R:3> {<R:5> <T:35> } {<R:7> <T:9> <T:111> <T:17> {<R:98> {<R:39> <T:111> {<R:100> <T:72> {<R:40> {<R:98> {<R:39> <T:111> } } } <T:71> } } } {<R:16> <T:63> {<R:18> {<R:4> {<R:5> <T:35> } } {<R:19> {<R:25> <T:111> {<R:42> <T:61> <T:62> } {<R:63> <T:63> {<R:64> {<R:67> {<R:82> {<R:81> <T:40> <T:61> {<R:80> {<R:82> {<R:86> {<R:22> {<R:98> {<R:39> <T:111> } } } <T:69> <T:9> } } } <T:62> } } <T:67> } } <T:64> } } } } <T:64> } } } <T:-1> }
```

[MIT License](LICENSE).
