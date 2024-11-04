#Generating text using Markov chains

#Start by importing the markovify library and a text file whose style we would like to imitate

import re
import markovify
import pandas as pd
df = pd.read_csv("assets/airport_reviews.csv")

# Next, join the individual reviews into one large text string and build a Markov chain model using the airport review text:

from itertools import chain
N = 100
review_subset = df["content"][0:N]
text = "".join(chain.from_iterable(review_subset))
markov_chain_model = markovify.Text(text)

#Generate five sentences using the Markov chain model:

for i in range(5):
 print(markov_chain_model.make_sentence())

#Generate 3 sentences with a length of no more than 140 characters:
print()
print()
for i in range(3):
    print(markov_chain_model.make_short_sentence(140))


class Text(object):
    reject_pat = re.compile(r"(^')|('$)|\s'|'\s|[\"(\(\)\[\])]")
    def __init__(self, input_text, state_size=2, chain=None, parsed_sentences=None, retain_original=True, well_formed=True, reject_reg=''):
        """
            input_text: A string.
            state_size: An integer, indicating the number of words in the
            model's state.
            chain: A trained markovify.Chain instance for this text, if preprocessed.
            parsed_sentences: A list of lists, where each outer list is a "run"
            of the process (e.g. a single sentence), and each inner list
            contains the steps (e.g. words) in the run. If you want to
            simulate
            an infinite process, you can come very close by passing just
            one, very
            long run.
            retain_original: Indicates whether to keep the original corpus.
            well_formed: Indicates whether sentences should be well-formed,
            preventing
            unmatched quotes, parenthesis by default, or a custom regular
            expression
            can be provided.
            reject_reg: If well_formed is True, this can be provided to
            override the
            standard rejection pattern.
        """
 
 