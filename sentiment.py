''' transformer example '''
from transformers import pipeline

if __name__ == "__main__":
    print("Learning transformers.")
    classifier = pipeline('sentiment-analysis')

    results = classifier([
     'lil lol lel',
     'this is very pleasing mate.',
     'have you ever heard of being pleased',
     'i disapprove this type of conversation.'
    ])
    for result in results:
        print(f"label: {result['label']}, with score {round(result['score'], 4)}")
