# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Python Resource Guide

Below are links to some recommended resources that past Data Science students have found helpful when learning Python. There is no single "best" resource; instead, you should try out a few and then focus on the one that teaches Python in a way that you understand.

* [Codecademy's Python course](http://www.codecademy.com/en/tracks/python): Good beginner material, including tons of in-browser exercises.
* [Dataquest](https://www.dataquest.io): Uses interactive exercises to teach Python in the context of data science.
* [Google's Python Class](https://developers.google.com/edu/python/): Slightly more advanced, and includes hours of useful lecture videos and downloadable exercises (with solutions).
* [Introduction to Python](http://introtopython.org/): A series of IPython notebooks that do a great job explaining core Python concepts and data structures.
* [Python for Informatics](http://www.pythonlearn.com/book.php): A very beginner-oriented book, with associated [slides](https://drive.google.com/folderview?id=0B7X1ycQalUnyal9yeUx3VW81VDg&usp=sharing) and [videos](https://www.youtube.com/playlist?list=PLlRFEj9H3Oj4JXIwMwN1_ss1Tk8wZShEJ).
* [A Crash Course in Python for Scientists](http://nbviewer.ipython.org/gist/rpmuller/5920182): Read through the Overview section for a very quick introduction to Python.
* [Python 2.7 Quick Reference](https://github.com/justmarkham/python-reference/blob/master/reference.py): A beginner-oriented guide that demonstrates Python concepts through short, well-commented examples.
* [Python Tutor](http://pythontutor.com/): Allows you to visualize the execution of Python code.

***

### Python Self-Assessment: Questions

1. How do you create an empty list named "mylist"?
2. What will the following code return? `5 > 3 or 5 < 3`
3. What will be stored in the "nums" object? `nums = range(10)`
4. How do you check the type and the length of the "nums" object?
5. How do you return the last number in the "nums" object?
6. Slice the "nums" object to return a list with the numbers 2, 3, 4.
7. What is the difference between `nums.append(10)` and `nums + [10]`?
8. How do you divide 3 by 2 and get a result of 1.5?
9. Import the "math" module, and then use its "sqrt" function to calculate the square root of 1089.
10. What type of object is created by this code? `d = {'a':10, 'b':20, 'c':30}`
11. In the "d" object, what are "a", "b", and "c" called? What are 10, 20, and 30 called?
12. How do you return the 10 from the "d" object?
13. How do you change the 30 to a 40 in the "d" object?
14. From the "people" object, return Brandon's state only: `people = {'Alice': ['Washington', 'DC'], 'Brandon': ['Arlington', 'VA']}`
15. Define a function "calc" that takes two variables, "a" and "b", and returns their sum.

***



### SPOILER ALERT
DON'T REVIEW UNTIL YOU'VE COMPLETED THE SELF-ASSESSMENT



<spoiler>
<details>
> ### Python Self-Assessment: Answers

> 1. `mylist = []` or `mylist = list()`
> 2. True
> 3. A list of the integers 0 through 9
> 4. `type(nums)` and `len(nums)`
> 5. `nums[9]` or `nums[-1]`
> 6. `nums[2:5]`
> 7. `nums.append(10)` modifies the original list, whereas `nums + [10]` does not actually modify the list.
> 8. In Python 3: `3/2`. In Python 2: `3/float(2)` or `3/2.0`.
> 9. `import math` and then `math.sqrt(1089)`. Alternatively, `from math import sqrt` and then `sqrt(1089)`.
> 10. A dictionary
> 11. "a", "b", and "c" are the keys, and 10, 20, and 30 are the values.
> 12. `d['a']`
> 13. `d['c'] = 40`
> 14. `people['Brandon'][1]`
> 15. `def calc(a, b): return a + b` (usually written as two separate lines)
</details>
</spoiler>
