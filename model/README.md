Project Models
==============

## n-gram

Uses [`kenlm`](https://github.com/kpu/kenlm) for ngrams.

```bash
# current directory is cmput563_insyn/..
$ sudo apt install libboost-all-dev cmake
$ git clone https://github.com/kpu/kenlm.git
$ cd kenlm
$ git checkout a1fcd16228eaf95edd36c83db4cc6ba1d65ed9d4 # optional
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j 4

# now feed in the training data tokens
$ ./insyn.py --tokenize-training-data | ../kenlm/build/bin/lmplz -o 10 > java-tokenstr-10grams.arpa
```

Trained model available on Cybera:
- https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/insyn/java-tokenstr-10grams.arpa.bin
  - 791.08 MB (faster loading)
- https://swift-yeg.cloud.cybera.ca:8080/v1/AUTH_e3b719b87453492086f32f5a66c427cf/insyn/java-tokenstr-10grams.arpa
  - 3.16 GB (show n-gram token sequence probabilities)
```bash
$ shasum -a 256 java-tokenstr-10grams.arpa
8534e8578905dc8449afc3dded9888fa747d2e86f55ce027ceb57ec276649af4  java-tokenstr-10grams.arpa
$ shasum -a 256 java-tokenstr-10grams.arpa.bin 
8b1aeb226af2159fac3b7914e50820851077b0c6202f2a2b5bc8f85078316ed3  java-tokenstr-10grams.arpa.bin
```

To allow python to run order 10 models:
- Ensure your python virtual environment is active, go to the kenlm directory
- Modify `setup.py` so that the argument `-DKENLM_MAX_ORDER=10`:
```text
$ git diff
diff --git a/setup.py b/setup.py
index 6193cf8..3dd6d15 100644
--- a/setup.py
+++ b/setup.py
@@ -21,7 +21,7 @@ else:
     LIBS = []
 
 #We don't need -std=c++11 but python seems to be compiled with it now.  https://github.com/kpu/kenlm/issues/86
-ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11']
+ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=10', '-std=c++11']
 
 #Attempted fix to https://github.com/kpu/kenlm/issues/186
 if platform.system() == 'Darwin':
```
- `pip install -e .`
