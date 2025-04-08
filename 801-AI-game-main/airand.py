
import random


def selectRowCol(boardDisplay):
    TotalRows=len(boardDisplay)
    TotalCols=len(boardDisplay[0])
    row = random.randint(0,TotalRows-1)
    col = random.randint(0,TotalCols-1)
    return int(row),int(col)