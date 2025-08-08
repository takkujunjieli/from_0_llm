import cProfile

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

cProfile.run('slow_function()', sort='cumtime')
