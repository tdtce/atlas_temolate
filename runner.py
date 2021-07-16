import concurrent.futures
import itertools
import multiprocessing
import warnings
from os import cpu_count
from typing import Any, Callable, List, Optional

from tqdm import tqdm


class Runner:
    def __init__(
        self, runner_type: str, num_workers: int, start_method: Optional[str] = None, progressbar: bool = True
    ):
        """
        A class for executing a function on a list of input parameters. Each input parameter can also represent an
        iterable, which will be substituted into the function using an asterisk (func(*params)).

        runner_type: Allows you to choose how to perform the computation loop: regular python loop,
        multithreading, or multiprocessing. Be careful when using with libraries that implicitly use
        multithreading / multiprocessing optimization as memory leaks are possible.
        num_workers: number of workers for 'thread' and 'process` runner type.
        start_method: option for multiprocessing allows you to specify 'fork' or 'spawn' method. If not specified,
        it is selected by default for the OS.

        NOTE:
            1) Errors are possible in 'process' mode when working with libraries using GPU (e.g., cv2).
            Change the start method to 'spawn', or use 'thread' runner_type.

            2) In 'process' mode, passed function must not be declared as nested within other functions:
            GOOD:
            def calculations(*args):
                return some_actions(*args)
            def dummy():
                runner = Runner('process', 5)
                runner(calculations, [1, 2, 3, 4])

            EXCEPTION (if you are lucky) or DEADLOCK:
            def dummy():
                runner = Runner('process', 5)
                def calculations(*args):
                    return some_actions(*args)
                runner(calculations, [1, 2, 3, 4])
        """

        assert isinstance(runner_type, str)
        assert runner_type in ["thread", "process", "loop"]

        assert num_workers > 0 and isinstance(num_workers, int)

        self.num_workers = min(num_workers, cpu_count())
        self.runner_type = runner_type

        if self.num_workers == 1 and self.runner_type != "loop":
            warnings.warn("Num workers is 1. Runner was changed to loop execution")
            self.runner_type = "loop"

        if self.runner_type == "process" and start_method is not None:
            assert start_method in ["spawn", "fork"]
            multiprocessing.set_start_method(start_method)

        self.progressbar = progressbar

    def __call__(self, function: Callable, params_list: List[Any]) -> List[Any]:
        return self.run(function, params_list)

    def run(self, function: Callable, params_list: List[Any]) -> List[Any]:
        """
        The general method for running a function on list of parameters.
        The method returns a list of results in the correct order (params_list[0] => results_list[0]).
        """
        if self.runner_type == "thread":
            return self._run_in_thread(function, params_list)
        elif self.runner_type == "process":
            return self._run_in_process(function, params_list)
        elif self.runner_type == "loop":
            return self._run_in_loop(function, params_list)
        else:
            raise NotImplementedError("Not implemented runner")

    def _run_in_loop(self, function: Callable, params_list: List[Any]) -> List[Any]:
        """
        Method execute function on list of parameters with python loop
        """
        if self.progressbar:
            params_list = tqdm(params_list, desc=f'Function "{function.__name__}" runned in "{self.runner_type}"')

        results_list = []
        for params in params_list:
            results_list.append(function(*params if isinstance(params, (list, tuple)) else [params]))
        return results_list

    def _run_in_thread(self, function: Callable, params_list: List[Any]) -> List[Any]:
        executor = concurrent.futures.ThreadPoolExecutor(self.num_workers)
        res = self._run_with_executor(executor, function, params_list)
        executor.shutdown()
        return res

    def _run_in_process(self, function: Callable, params_list: List[Any]) -> List[Any]:
        executor = concurrent.futures.ProcessPoolExecutor(self.num_workers)
        return self._run_with_executor(executor, function, params_list)

    def _run_with_executor(
        self, specific_executor: concurrent.futures.Executor, function: Callable, params_list: List[Any]
    ) -> List[Any]:
        """
        Method execute function on list of parameters in multithread or multiprocess mode
        """
        if self.progressbar:
            progressbar = tqdm(
                total=len(params_list),
                desc=f'Function "{function.__name__}" runned in "{self.num_workers} {self.runner_type}"',
            )
        else:
            progressbar = None

        params_list_iterator = iter(params_list)

        # The result of the task becomes available in a different order than the tasks are declared.
        # After completing all the tasks, they must be sorted in the order of their declaration.
        results_dict = {}
        idx = 0
        with specific_executor as executor:
            # Schedule the first N futures.  We don't want to schedule them all
            # at once, to avoid consuming excessive amounts of memory.
            futures = {}
            fut2idx = {}
            for params in itertools.islice(params_list_iterator, self.num_workers):
                fut = executor.submit(function, *params if isinstance(params, (list, tuple)) else [params])
                futures[fut] = params
                fut2idx[fut] = idx
                idx += 1

            while futures:
                # Wait for the next future to complete.
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    del futures[fut]
                    results_dict[fut2idx[fut]] = fut.result()
                    del fut

                if progressbar:
                    progressbar.update(idx - progressbar.n)

                # Schedule the next set of futures.  We don't want more than N futures
                # in the pool at a time, to keep memory consumption down.
                for params in itertools.islice(params_list_iterator, len(done)):
                    fut = executor.submit(function, *params if isinstance(params, (list, tuple)) else [params])
                    futures[fut] = params
                    fut2idx[fut] = idx
                    idx += 1

        # Return the results as a list, ordered according to the task declaration
        results_list = []
        for key in sorted(results_dict.keys()):
            results_list.append(results_dict[key])

        del results_dict, futures, fut2idx, params_list_iterator, params_list
        return results_list