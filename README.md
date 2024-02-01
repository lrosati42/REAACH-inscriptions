This is the official implementation of the processing pipeline described in the paper 
### "Between image and text: automatic image processing for character recognition in historical inscriptions".


The code is tested to work in Linux with Python 3.12.1 and the minimal libraries necessary to run the code can be installed creating a conda environment from the file _environment.yml_

To run the processing pipeline on a given .obj point cloud file, run 
```
python main -d [name of the .obj file directory] -f [filename without extension] -o [directory of the output file]
```

For example, you can run 

```
python main.py -d data/random/clouds/ -f r6 -o out/
```
to process the file r6.obj included in the data folder.

The same command can be given in a more explicit version by running
```
python main.py --directory data/random/clouds/ --file r6 --output out/
```
