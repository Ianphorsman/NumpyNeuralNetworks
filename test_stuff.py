# original array



def mapSequenceDirection(arr):
    directionMap = []
    i = 0
    while i < len(arr)-1:
        directionMap.append(arr[i] > arr[i+1])
        i += 1
    return directionMap

def isIncreasingSequence(directionMap, orig_arr):
    numTrues = directionMap.count(True)
    numFalses = directionMap.count(False)
    if (numTrues == 1 and numFalses >= 1):
        orig_arr.pop(directionMap.index(numTrues)+1)
        new_seq = mapSequenceDirection(orig_arr)
    elif (numFalses == 1 and numTrues >= 1):
        orig_arr.pop(directionMap.index(numFalses)+1)
    else:
        return False


test_arrays = [
    [1, 4, 10, 4, 2],
    [1, 1],
    [1, 2, 3, 4, 5, 3, 5, 6],
    [3, 5, 67, 98, 3]
]

for test_arr in test_arrays:
    print(mapSequenceDirection(test_arr), "Verdict: ", isIncreasingSequence(mapSequenceDirection(test_arr), test_arr))