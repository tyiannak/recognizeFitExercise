# recognizeFitExercise
Classification of fitness exercises based on accelerometer and camera information

# Usage:
## Evaluate only-accelerometer classification
Only-X:
```
python accelerometer.py classifier Accelerometer\ Data/ 1 0
```

Only-Y:
```
python accelerometer.py classifier Accelerometer\ Data/ 2 0
```

Only-Z:
```
python accelerometer.py classifier Accelerometer\ Data/ 3 0
```

All accelerometer axes:
```
python accelerometer.py classifier Accelerometer\ Data/ 4 0
```


## Evaluate camera-only classification
```
python accelerometer.py classifier Accelerometer\ Data/ 0 1
```


## Evaluate fusion classification
```
python accelerometer.py classifier Accelerometer\ Data/ 4 1
```
