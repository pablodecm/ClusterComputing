from IPython.parallel import Client

rc = Client()
dview = rc[:]
lview = rc.load_balanced_view()

with dview.sync_imports():
    import numpy
    from sklearn import svm

import sys
import time
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import datasets
from sklearn import svm

# We use a decorator here, to give us a nicer syntax for calling map.
# block=False means svm_params_crossval.map will return an AsyncResult
# chunksize=1 means that each engine will only receive one bit of a data at a time.
# You will want to play with this setting to see what gives you the best results.
@lview.parallel(block=False, chunksize=1)
def svm_params_crossval(indexes):
    train_idx, crossval_idx = indexes
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_crossval = X[crossval_idx]
    y_crossval = y[crossval_idx]
    crossval_err = numpy.zeros((C_range.size, gamma_range.size))
    for i, C in enumerate(C_range):
        for j, gamma in enumerate(gamma_range):
            clf = svm.SVC(C=C, gamma=gamma)
            clf.fit(X_train, y_train)
            crossval_err[i, j] = 1. - clf.score(X_crossval, y_crossval)
    return crossval_err


def short_format_time(t):
    if t > 60:
        return "%4.1fmin" % (t / 60.)
    else:
        return " %5.1fs" % t


def wait_progress(ar, interval=5, timeout=-1):
    """Wait on an IPython AsyncResult, printing progress to stdout.

    Based on wait_interactive() in IPython and the output of Joblib in verbose mode.parallel

    This will work best when using a load-balanced view with a smallish chunk-size.

    """
    if timeout is None:
        timeout = -1
    N = len(ar)
    tic = time.time()
    print "\nRunning %i tasks:" % N
    sys.stdout.flush()
    last = 0
    while not ar.ready() and (timeout < 0 or time.time() - tic <= timeout):
        ar.wait(interval)
        progress, elapsed = ar.progress, ar.elapsed
        if progress > last:
            last = progress
            remaining = elapsed * (float(N) / progress - 1.)
            print '    Done %4i out of %4i | elapsed: %s remaining: %s' % (
                progress, N, short_format_time(elapsed), short_format_time(remaining))
            sys.stdout.flush()
    if ar.ready():
        try:
            speedup = round(100.0 * ar.serial_time / ar.wall_time)
            print "\nParallel speedup: %i%%" % speedup
        # For some reason ar.serial_time occasionally throws this exception. 
        # We choose to ignore it and just not display the speedup factor.
        except TypeError:
            pass


def main():
    # Load a "toy" data set
    iris = datasets.load_iris()
    X = preprocessing.scale(iris.data)
    y = iris.target

    # Set the range hyperparameters we want to search
    C_range = 10. ** numpy.arange(-2, 9)
    gamma_range = 10. ** numpy.arange(-5, 4)

    # Send out the data to the engines via the direct view
    dview.push(dict(X=X, y=y, C_range=C_range, gamma_range=gamma_range), block=True)

    # Run svm_params_crossval in parallel. Note the nice syntax afforded by using
    # the @lview.parallel decorator. This is equivalent to:
    # ar = lview.map_async(svm_params_crossval, cross_validation.LeaveOneOut(len(y)), chunksize=1)
    ar = svm_params_crossval.map(cross_validation.LeaveOneOut(len(y)))
    try:
        # Busy waiting on results, to give nice progress updates
        wait_progress(ar)
    # Handle ctrl-c by aborting jobs before exiting. If we didn't do this, the tasks would
    # keep running to completion.
    except KeyboardInterrupt:
        print "Aborting..."
        sys.stdout.flush()
        ar.abort()
        sys.exit()

    # get the actual results
    results = ar.result
    # Average the results and convert to percent
    crossval_err = 100. * numpy.mean(results, axis=0)

    # find the C and gamma that gave us the lowest average cross-validation error
    min_idx = crossval_err.argmin()
    C_idx, gamma_idx = numpy.unravel_index(min_idx, crossval_err.shape)
    C_best = C_range[C_idx]
    gamma_best = gamma_range[gamma_idx]
    err_best = crossval_err[C_idx, gamma_idx]

    print "\nBest: C = %s, gamma = %s, err = %s%%\n" % (C_best, gamma_best, err_best)
    numpy.set_printoptions(precision=2, linewidth=120)
    print crossval_err


if __name__ == '__main__':
    # Track the overall time of computation
    start_time = time.time()
    main()
    end_time = time.time()
    print "\nTotal time: %s" % short_format_time(end_time - start_time)