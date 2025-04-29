from collections import namedtuple
import time
import pandas as pd
from itertools import combinations
import streamlit as st

# https://discord.com/channels/1028502733487079444/1031993265811095622
# https://claude.ai/chat/849f767b-7d22-4bcb-b3f1-a195b8740271

#Here's a short code sample I came up with to help illustrate how itertools is useful even for mostly analytical work.

# The below code sketches out how one might try to optimise some data analysis to find the best pipeline of preprocessing steps that minimising some cost (under the assumption that some preprocessing steps may be very costly but may not actually material improve our analysis.)

from collections import namedtuple
import time
import pandas as pd
from itertools import combinations

# Define the timed context manager
class timed:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.total_time = time.time() - self.start
    
    @property
    def total_time(self):
        return self._total_time
    
    @total_time.setter
    def total_time(self, value):
        self._total_time = value

def powerset(it):
    return (
        c
        for r in range(len(it) + 1)
        for c in combinations(it, r)
    )

Result = namedtuple('Result', 'result costs')

class Preprocess(namedtuple('Preprocess', 'description callable')):
    def __call__(self, *args, **kwargs):
        with timed() as t:
            rv = self.callable(*args, **kwargs)
        return Result(result=rv, costs=[t.total_time])

class Pipeline(namedtuple('Pipeline', 'steps')):
    def __call__(self, *args, **kwargs):
        costs = []
        result = None
        for idx, st in enumerate(self.steps, start=1):
            rv = st(*args, **kwargs) if idx == 1 else st(result)
            costs.extend(rv.costs)
            result = rv.result
        return Result(result=result, costs=costs)

if __name__ == '__main__':
    # Define sample data (a list of numbers)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Define preprocessing steps
    def remove_outliers(data_list):
        """Remove values that are more than 1 standard deviations from the mean"""
        if not data_list:
            return data_list
        s = pd.Series(data_list)
        mean, std = s.mean(), s.std()
        return list(s[(s > mean - 1*std) & (s < mean + 1*std)])
    
    def normalize(data_list):
        """Normalize values to range [0,1]"""
        if not data_list:
            return data_list
        min_val, max_val = min(data_list), max(data_list)
        if min_val == max_val:
            return [0.5] * len(data_list)
        return [(x - min_val) / (max_val - min_val) for x in data_list]
    
    def square_values(data_list):
        """Square all values"""
        return [x**2 for x in data_list]
    
    preprocessing = {
        Preprocess('Remove Outliers', remove_outliers),
        Preprocess('Normalize', normalize),
        Preprocess('Square Values', square_values)
    }
    
    process_1=Preprocess('Remove Outliers', square_values)
    result_1=process_1(data)
    st.write('this is the result 1', result_1.result)

    # Generate all possible pipeline combinations
    pipelines = {Pipeline(steps) for steps in powerset(preprocessing)}
    
    # Define scoring function
    def score(result):
        """Calculate a simple score (sum of values)"""
        return sum(result) if result else 0
    
    # Run all pipelines and get results
    results = {pl: pl(data) for pl in sorted(pipelines, key=lambda p: len(p.steps))}
    
    # Create DataFrame with results
    df = pd.DataFrame(
        index=[', '.join(step.description for step in pl.steps) or 'No steps' for pl in results.keys()],
        data={
            'score': [score(res.result) for res in results.values()],
            'cost': [sum(res.costs) for res in results.values()],
        },
    )
    
    st.write("Data:", data)
    st.write("\nPipeline Results:")
    st.write(df.applymap(lambda x: f"{float(x):.6f}")) # https://claude.ai/chat/e27d1ac2-b777-4412-b2ef-534863660105

    
    
    # Print example of processed data for a specific pipeline
    st.write("\nExample of processed data:")
    for name, result in zip(df.index, results.values()):
        # st.write('the below works')
        # st.write(f"this is name: {name}: this is the result: {result.result}")
        # st.write('now i want to clean up the formating below:')
        # Check if result is None or empty before formatting
        if result.result is None:
            formatted_result = "None"
        else:
            try:
                formatted_result = [f"{value:.2f}" for value in result.result]
            except (TypeError, ValueError):
                # Handle case where values might not be formattable as floats
                formatted_result = str(result.result)
        
        st.write(f"{name}: {formatted_result}")
        # st.write()