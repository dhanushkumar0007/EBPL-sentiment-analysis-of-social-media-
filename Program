from transformers import pipeline
import matplotlib.pyplot as plt

# Load the emotion classification pipeline
emotion_classifier = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)

# Social media texts
texts = [
    "I am so happy with my new phone!",
    "This is the worst movie I've ever seen.",
    "I’m scared of what might happen tomorrow.",
    "Wow, I didn't expect that! Amazing!",
    "Feeling a bit down today."
]

# Analyze and plot top 3 emotions for each text
for idx, text in enumerate(texts):
    emotions = emotion_classifier(text)[0]
    emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:3]

    labels = [e['label'] for e in emotions]
    scores = [e['score'] for e in emotions]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color='skyblue')
    plt.title(f"Top Emotions for Text {idx+1}")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()
