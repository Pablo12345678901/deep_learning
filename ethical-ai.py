import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp

###########################################################
############# BEGINNING OF LARGE DATA #####################
###########################################################
# WARNING : large data below (~400 lines)
# Scenarios on which train the AI
# Training data into json format - could be enhanced by loading from external file instead of keeping such large data (in code space equivalence).
training_scenarios = [
    {
        "description": "A self-driving car must choose between hitting a group of pedestrians or swerving and risking the life of its passenger.",
        "actions": [
            {"description": "Protect the passenger", "harm_reduction": 1, "responsibility": 1},
            {"description": "Protect the pedestrians", "fairness": 1, "beneficence": 1}
        ]
    },
    {
        "description": "A doctor must decide whether to give a limited supply of medicine to one critically ill patient or distribute it among five patients with moderate illnesses.",
        "actions": [
            {"description": "Treat the critically ill patient", "harm_reduction": 1, "fairness": 0},
            {"description": "Treat the five moderate patients", "fairness": 1, "equality": 1}
        ]
    },
    {
        "description": "A company must choose between maximizing profits and reducing environmental impact.",
        "actions": [
            {"description": "Maximize profits", "responsibility": 0, "sustainability": 0},
            {"description": "Reduce environmental impact", "sustainability": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A school must decide whether to implement affirmative action in its admissions process.",
        "actions": [
            {"description": "Implement affirmative action", "fairness": 1, "equality": 1},
            {"description": "Maintain current admissions process", "fairness": 0, "equality": 0}
        ]
    },
    {
        "description": "A government must decide whether to implement mass surveillance for national security.",
        "actions": [
            {"description": "Implement mass surveillance", "harm_reduction": 1, "autonomy": 0},
            {"description": "Protect privacy", "autonomy": 1, "dignity": 1}
        ]
    },
    {
        "description": "A doctor must decide whether to perform a risky surgery that could save a patient's life.",
        "actions": [
            {"description": "Perform the surgery", "beneficence": 1, "harm_reduction": 1},
            {"description": "Don't perform the surgery", "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to release a product that could be harmful to the environment.",
        "actions": [
            {"description": "Release the product", "responsibility": 0, "sustainability": 0},
            {"description": "Don't release the product", "responsibility": 1, "sustainability": 1}
        ]
    },
    {
        "description": "A government must decide whether to provide aid to a country with a history of human rights abuses.",
        "actions": [
            {"description": "Provide aid", "beneficence": 1, "responsibility": 0},
            {"description": "Don't provide aid", "justice": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to use child labor in a developing country to reduce costs.",
        "actions": [
            {"description": "Use child labor", "fairness": 0, "equality": 0},
            {"description": "Don't use child labor", "fairness": 1, "equality": 1}
        ]
    },
    {
        "description": "A government must decide whether to censor certain information for national security reasons.",
        "actions": [
            {"description": "Censor the information", "harm_reduction": 1, "transparency": 0},
            {"description": "Don't censor the information", "transparency": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company has developed an AI that can predict criminal behavior. They must decide whether to sell it to law enforcement agencies.",
        "actions": [
            {"description": "Sell the AI", "harm_reduction": 1, "responsibility": 0},
            {"description": "Don't sell the AI", "autonomy": 1, "dignity": 1}
        ]
    },
    {
        "description": "A government must decide whether to use torture to extract information from a suspected terrorist.",
        "actions": [
            {"description": "Use torture", "harm_reduction": 1, "responsibility": 0},
            {"description": "Don't use torture", "dignity": 1, "justice": 1}
        ]
    },
    {
        "description": "A company must decide whether to use genetic testing to screen job applicants for certain traits.",
        "actions": [
            {"description": "Use genetic testing", "fairness": 0, "equality": 0},
            {"description": "Don't use genetic testing", "privacy": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A government must decide whether to develop a powerful new weapon of mass destruction.",
        "actions": [
            {"description": "Develop the weapon", "harm_reduction": 0, "responsibility": 0},
            {"description": "Don't develop the weapon", "harm_reduction": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to dump toxic waste in a developing country with lax environmental regulations.",
        "actions": [
            {"description": "Dump the waste", "responsibility": 0, "sustainability": 0},
            {"description": "Don't dump the waste", "responsibility": 1, "sustainability": 1}
        ]
    },
    {
        "description": "A government must decide whether to provide aid to a country that has recently experienced a natural disaster.",
        "actions": [
            {"description": "Provide aid", "beneficence": 1, "responsibility": 1},
            {"description": "Don't provide aid", "beneficence": 0, "responsibility": 0}
        ]
    },
    {
        "description": "A company must decide whether to automate jobs, potentially causing unemployment.",
        "actions": [
            {"description": "Automate jobs", "efficiency": 1, "responsibility": 0},
            {"description": "Maintain human jobs", "responsibility": 1, "fairness": 1}
        ]
    },
    {
        "description": "A government must decide whether to legalize euthanasia for terminally ill patients.",
        "actions": [
            {"description": "Legalize euthanasia", "autonomy": 1, "harm_reduction": 1},
            {"description": "Maintain ban on euthanasia", "sanctity_of_life": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to use predictive algorithms for hiring decisions.",
        "actions": [
            {"description": "Use predictive algorithms", "efficiency": 1, "fairness": 0},
            {"description": "Rely on human judgment", "fairness": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income.",
        "actions": [
            {"description": "Implement universal basic income", "equality": 1, "beneficence": 1},
            {"description": "Maintain current welfare system", "responsibility": 1, "efficiency": 1}
        ]
    },
        {
        "description": "A company must decide whether to collect and sell customer data for targeted advertising.",
        "actions": [
            {"description": "Collect and sell customer data", "profits": 1, "responsibility": 0},
            {"description": "Protect customer privacy", "privacy": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow gene editing of human embryos.",
        "actions": [
            {"description": "Allow gene editing", "harm_reduction": 1, "beneficence": 1},
            {"description": "Ban gene editing", "sanctity_of_life": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI assistant that can write content for customers.",
        "actions": [
            {"description": "Develop the AI assistant", "efficiency": 1, "profits": 1},
            {"description": "Maintain human-written content", "quality": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a carbon tax to reduce emissions.",
        "actions": [
            {"description": "Implement a carbon tax", "sustainability": 1, "responsibility": 1},
            {"description": "Maintain current emissions policies", "efficiency": 1, "profits": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously make business decisions.",
        "actions": [
            {"description": "Develop the autonomous AI system", "efficiency": 1, "profits": 1},
            {"description": "Maintain human decision-making", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow the use of psychedelic drugs for therapeutic purposes.",
        "actions": [
            {"description": "Allow therapeutic use of psychedelics", "harm_reduction": 1, "autonomy": 1},
            {"description": "Maintain prohibition on psychedelics", "public_safety": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop a social media platform that uses AI to curate content.",
        "actions": [
            {"description": "Develop the AI-curated platform", "engagement": 1, "profits": 1},
            {"description": "Maintain human-curated content", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal healthcare system.",
        "actions": [
            {"description": "Implement universal healthcare", "equality": 1, "beneficence": 1},
            {"description": "Maintain private healthcare system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage supply chains.",
        "actions": [
            {"description": "Develop the autonomous supply chain AI", "efficiency": 1, "profits": 1},
            {"description": "Maintain human supply chain management", "responsibility": 1, "resilience": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow the development of genetically modified organisms (GMOs) for agriculture.",
        "actions": [
            {"description": "Allow GMO development", "productivity": 1, "sustainability": 1},
            {"description": "Maintain restrictions on GMOs", "public_safety": 1, "responsibility": 1}
        ]
    },
        {
        "description": "A company must decide whether to collect and sell customer data for targeted advertising.",
        "actions": [
            {"description": "Collect and sell customer data", "profits": 1, "responsibility": 0},
            {"description": "Protect customer privacy", "privacy": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow gene editing of human embryos.",
        "actions": [
            {"description": "Allow gene editing", "harm_reduction": 1, "beneficence": 1},
            {"description": "Ban gene editing", "sanctity_of_life": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI assistant that can write content for customers.",
        "actions": [
            {"description": "Develop the AI assistant", "efficiency": 1, "profits": 1},
            {"description": "Maintain human-written content", "quality": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a carbon tax to reduce emissions.",
        "actions": [
            {"description": "Implement a carbon tax", "sustainability": 1, "responsibility": 1},
            {"description": "Maintain current emissions policies", "efficiency": 1, "profits": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously make business decisions.",
        "actions": [
            {"description": "Develop the autonomous AI system", "efficiency": 1, "profits": 1},
            {"description": "Maintain human decision-making", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow the use of psychedelic drugs for therapeutic purposes.",
        "actions": [
            {"description": "Allow therapeutic use of psychedelics", "harm_reduction": 1, "autonomy": 1},
            {"description": "Maintain prohibition on psychedelics", "public_safety": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop a social media platform that uses AI to curate content.",
        "actions": [
            {"description": "Develop the AI-curated platform", "engagement": 1, "profits": 1},
            {"description": "Maintain human-curated content", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal healthcare system.",
        "actions": [
            {"description": "Implement universal healthcare", "equality": 1, "beneficence": 1},
            {"description": "Maintain private healthcare system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage supply chains.",
        "actions": [
            {"description": "Develop the autonomous supply chain AI", "efficiency": 1, "profits": 1},
            {"description": "Maintain human supply chain management", "responsibility": 1, "resilience": 1}
        ]
    },
    {
        "description": "A government must decide whether to allow the development of genetically modified organisms (GMOs) for agriculture.",
        "actions": [
            {"description": "Allow GMO development", "productivity": 1, "sustainability": 1},
            {"description": "Maintain restrictions on GMOs", "public_safety": 1, "responsibility": 1}
        ]
    },
        {
        "description": "A company must decide whether to develop an AI system that can autonomously manage customer service.",
        "actions": [
            {"description": "Develop the autonomous customer service AI", "efficiency": 1, "costs": 1},
            {"description": "Maintain human customer service", "quality": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic education system.",
        "actions": [
            {"description": "Implement universal basic education", "equality": 1, "opportunity": 1},
            {"description": "Maintain current education system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage human resources.",
        "actions": [
            {"description": "Develop the autonomous HR AI", "efficiency": 1, "fairness": 1},
            {"description": "Maintain human HR management", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the elderly.",
        "actions": [
            {"description": "Implement universal basic income for the elderly", "equality": 1, "dignity": 1},
            {"description": "Maintain current retirement system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage financial investments.",
        "actions": [
            {"description": "Develop the autonomous investment AI", "efficiency": 1, "profits": 1},
            {"description": "Maintain human investment management", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the disabled.",
        "actions": [
            {"description": "Implement universal basic income for the disabled", "equality": 1, "beneficence": 1},
            {"description": "Maintain current disability support system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage legal compliance.",
        "actions": [
            {"description": "Develop the autonomous legal compliance AI", "efficiency": 1, "costs": 1},
            {"description": "Maintain human legal compliance management", "responsibility": 1, "trust": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the unemployed.",
        "actions": [
            {"description": "Implement universal basic income for the unemployed", "equality": 1, "opportunity": 1},
            {"description": "Maintain current unemployment support system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage cybersecurity.",
        "actions": [
            {"description": "Develop the autonomous cybersecurity AI", "efficiency": 1, "costs": 1},
            {"description": "Maintain human cybersecurity management", "responsibility": 1, "resilience": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the homeless.",
        "actions": [
            {"description": "Implement universal basic income for the homeless", "equality": 1, "beneficence": 1},
            {"description": "Maintain current homelessness support system", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
        {
        "description": "A company must decide whether to develop an AI system that can autonomously manage product design.",
        "actions": [
            {"description": "Develop the autonomous product design AI", "efficiency": 1, "innovation": 1},
            {"description": "Maintain human product design", "creativity": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the elderly and disabled.",
        "actions": [
            {"description": "Implement universal basic income for the elderly and disabled", "equality": 1, "dignity": 1},
            {"description": "Maintain current support systems", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage research and development.",
        "actions": [
            {"description": "Develop the autonomous R&D AI", "efficiency": 1, "innovation": 1},
            {"description": "Maintain human R&D management", "creativity": 1, "responsibility": 1}
        ]
    },
    {
        "description": "A government must decide whether to implement a universal basic income for the unemployed and homeless.",
        "actions": [
            {"description": "Implement universal basic income for the unemployed and homeless", "equality": 1, "opportunity": 1},
            {"description": "Maintain current support systems", "personal_responsibility": 1, "efficiency": 1}
        ]
    },
    {
        "description": "A company must decide whether to develop an AI system that can autonomously manage marketing and advertising.",
        "actions": [
            {"description": "Develop the autonomous marketing AI", "efficiency": 1, "engagement": 1},
            {"description": "Maintain human marketing management", "creativity": 1, "trust": 1}
        ]
    }
]


###########################################################
################### END OF LARGE DATA #####################
###########################################################

# Define a custom vectorizer that allows for word weighting
class WeightedCountVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word_weights=None):
        self.word_weights = word_weights or {}
        self.vectorizer = CountVectorizer()

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def fit_transform(self, X, y=None):
        X_transformed = self.vectorizer.fit_transform(X)
        
        # Convert the sparse matrix to a dense matrix of float64 type
        X_transformed = X_transformed.astype(float)
        
        if self.word_weights:
            for word, weight in self.word_weights.items():
                if word in self.vectorizer.vocabulary_:
                    word_idx = self.vectorizer.vocabulary_[word]
                    X_transformed[:, word_idx] *= weight
            
            # Convert the modified dense matrix back to a sparse matrix
            X_transformed = sp.csr_matrix(X_transformed)
        
        return X_transformed

    def transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        
        # Convert the sparse matrix to a dense matrix of float64 type
        X_transformed = X_transformed.astype(float)
        
        if self.word_weights:
            for word, weight in self.word_weights.items():
                if word in self.vectorizer.vocabulary_:
                    word_idx = self.vectorizer.vocabulary_[word]
                    X_transformed[:, word_idx] *= weight
            
            # Convert the modified dense matrix back to a sparse matrix
            X_transformed = sp.csr_matrix(X_transformed)
        
        return X_transformed

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


class EthicalLanguageAI:
    def __init__(self):
        # Define word weights for the vectorizer
        self.word_weights = {
            'death': 2.0,
            'life': 1.8,
            'harm': 1.7,
            'benefit': 1.5,
            'fair': 1.6,
            'unfair': 1.6,
            'choice': 1.4,
            'force': 1.5,
            'equal': 1.5,
            'discriminate': 1.6
        }
    
        # Define additional ethical values related to environmental protection
        # These values will be tested to see if the given action prioritizes environmental protection
        self.environmental_values = {
            "sustainability": 0.9,
            "climate_action": 0.9,
            "eco-justice": 0.8,
            "environmental_stewardship": 0.8,
            "conservation": 0.8,
            "preservation": 0.8,
            "biodiversity_protection": 0.8,
            "habitat_conservation": 0.8,
            "eco-friendliness": 0.8,
            "sustainable_development": 0.8,
            "renewable_energy": 0.8,
            "eco-justice": 0.8,
            "circular_economy": 0.7,
            "green_economy": 0.7,
            "ecological_balance": 0.7,
            "environmental_ethics": 0.7,
            "environmental_justice": 0.7,
            "environmental_activism": 0.7,
            "energy_efficiency": 0.7,
            "waste_reduction": 0.7,
            "restoration": 0.7,
            "regeneration": 0.7,
            "recycling": 0.6,
            "eco-spirituality": 0.6,
            "eco-feminism": 0.6,
            "eco-socialism": 0.6,
            "environmental_education": 0.6,
            "eco-tourism": 0.6,
            "biomimicry": 0.6,
            "permaculture": 0.6,
            "agroecology": 0.6,
            "eco-villages": 0.5
        }

        # Define ethical values and their importance weights
        self.ethical_values = {
            "compassion": 0.9,
            "honesty": 0.9,
            "sustainability": 0.9,
            "fairness": 0.8,
            "integrity": 0.8,
            "justice": 0.8,
            "equality": 0.8,
            "stewardship": 0.8,
            "responsibility": 0.7,
            "kindness": 0.8,
            "beneficence": 0.9,
            "non-violence": 0.8,
            "trust": 0.7,
            "community": 0.7,
            "generosity": 0.7,
            "charity": 0.7,
            "humanism": 0.7,
            "altruism": 0.7,
            "freedom": 0.7,
            "dignity": 0.7,
            "accountability": 0.7,
            "resilience": 0.7,
            "innovation": 0.7,
            "creativity": 0.7,
            "excellence": 0.7,
            "empathy": 0.7,
            "wisdom": 0.7,
            "peace": 0.7,
            "truthfulness": 0.7,
            "diversity": 0.7,
            "inclusion": 0.7,
            "respect": 0.7,
            "autonomy": 0.6,
            "courage": 0.6,
            "humility": 0.6,
            "temperance": 0.6,
            "forgiveness": 0.6,
            "loyalty": 0.6,
            "diligence": 0.6,
            "perseverance": 0.6,
            "adaptability": 0.6,
            "fulfillment": 0.6,
            "meaning": 0.6,
            "purpose": 0.6,
            "service": 0.6,
            "harmony": 0.6,
            "privacy": 0.6,
            "transparency": 0.6,
            "self-control": 0.5,
            "moderation": 0.5,
            "contentment": 0.5,
            "simplicity": 0.5,
            "gratitude": 0.5,
            "patience": 0.5,
            "transcendence": 0.5,
            "spirituality": 0.5,
            "enlightenment": 0.5
        }
        self.vectorizer = WeightedCountVectorizer(word_weights=self.word_weights)
        self.classifier = MultinomialNB()
        self.scenario_count = 1  # Initialize scenario count
        self.category_labels = []  # This will store our category labels

    def train(self, scenarios, labels):
        # Extract descriptions from scenarios
        descriptions = [scenario['description'] for scenario in scenarios]
        
        # Transform the descriptions using the weighted vectorizer
        X = self.vectorizer.fit_transform(descriptions).toarray()
        
        # Extract unique labels from ethical_values and environmental_values
        self.category_labels = list(set(self.ethical_values.keys()) | set(self.environmental_values.keys()))
        
        # Create labels based on the unique values in the actions
        y = []
        for scenario in scenarios:
            label_vector = [0] * len(self.category_labels)
            for action in scenario['actions']:
                for label in self.category_labels:
                    if label in action:
                        label_vector[self.category_labels.index(label)] += 1
            y.append(np.argmax(label_vector))
        y = np.array(y)
        
        # Train the classifier on the transformed data
        self.classifier.fit(X, y)

    def analyze_scenario(self, scenario):
        # Transform the scenario description
        X = self.vectorizer.transform([scenario['description']]).toarray()
        
        # Predict the ethical category of the scenario
        predicted_category = self.classifier.predict(X)[0]
        return predicted_category

    def evaluate_action(self, action, scenario_category):
        score = 0
        
        for value, weight in self.environmental_values.items():
            if value in action:
                score += weight * action[value]
        
        for value, weight in self.ethical_values.items():
            if value in action:
                score += weight * action[value]
        
        if scenario_category in action:
            score += 0.5 * action[scenario_category]

        return score

    def decide(self, scenario):
        # Analyze the scenario to determine its ethical category
        category_index = self.analyze_scenario(scenario)
        category_label = self.category_labels[category_index]
        
        # Print scenario number and description
        output = f"Test Scenario {self.scenario_count}: \n{scenario['description']}\n\n"

        # Print the predicted category number and its label
        output += f"Predicted category: {category_label} - {category_index}\n\n"

        # Evaluate each possible action and print choices with scores
        output += "Choices available:\n"
        for action in scenario["actions"]:
            score = self.evaluate_action(action, category_index)
            output += f"- {action['description']} (Score: {score:.2f})\n"
        output += "\n"  # Add a blank line after choices
        
        # Store the final best_action as the choice of the AI
        best_action = max(scenario["actions"], key=lambda x: self.evaluate_action(x, category_index))
        best_score = self.evaluate_action(best_action, category_index)
        ai_decision = best_action

        # Define the percentage and label tuples
        percentage_labels = [
            ( 0, "Completely Unethical"),
            (10, "Somewhat Unethical"),
            (20, "Neutral"),
            (40, "Somewhat Ethical"),
            (50, "Ethical"),
            (60, "Very Ethical"),
            (70, "Highly Ethical"),
            (80, "Extremely Ethical"),
            (90, "Perfectly Ethical"),
            (100, "Exceptionally Ethical")
        ]

        # Calculate the percentage and label for the AI's decision
        ai_decision_score = self.evaluate_action(ai_decision, category_index)
        ai_decision_percentage = ai_decision_score / 4 * 100
        ai_decision_label = self.get_label(ai_decision_score, percentage_labels)

        # Calculate the percentage and label for the best decision
        best_decision_percentage = best_score / 4 * 100
        best_decision_label = self.get_label(best_score, percentage_labels)

        output += "AI's decision VS Best decision available:\n"
        output += f"AI's : {ai_decision_percentage:.2f}% {ai_decision_label} {ai_decision_score:.2f} {ai_decision['description']}\n"
        output += f"Best : {best_decision_percentage:.2f}% {best_decision_label} {best_score:.2f} {best_action['description']}\n"
        output += "________________________________________________________________\n"
        
        print(output)
        self.scenario_count += 1  # Increment scenario count

        return best_action, best_score, category_index

    def get_label(self, score, percentage_labels):
        for percentage, label in percentage_labels:
            if score <= percentage * 0.01 * 4:
                return label
        return percentage_labels[-1][1]  # Return the last label if score exceeds all thresholds
    

# Create a list of unique ethical values from the training scenarios
unique_values = set()
for scenario in training_scenarios:
    for action in scenario['actions']:
        unique_values.update(action.keys())

# Convert the list of unique values to a list of labels
training_labels = list(unique_values)

# Create and Train the AI
ai = EthicalLanguageAI()
ai.train(training_scenarios, training_labels)

# Test the AI on the given test scenarios and provide a decision including a printed result.
for i, scenario in enumerate(training_scenarios, 1):
    decision, score, category = ai.decide(scenario)

# Simulate multiple decisions for a given scenario
def simulate_decisions(ai, scenario, num_simulations):
    decisions = {'Protect the passenger': 0, 'Protect the pedestrians': 0}
    for _ in range(num_simulations):
        decision, _, _ = ai.decide(scenario)
        decisions[decision['description']] += 1
    return decisions

# Run a simulation on the first test scenario
simulation_scenario = training_scenarios[0]
num_simulations = 1000

# Get the AI's decision for each test scenario
ai = EthicalLanguageAI()
ai.train(training_scenarios, training_labels)


# Conclusion
print("================================================================")
print("Conclusion :")
print("================================================================\n")
print("This ethical AI demonstrates the complexity of making moral decisions.")
print("It considers multiple ethical values and uses natural language processing to analyze scenarios. However, it's important to note that this is a simplified model and real-world ethical decision-making is far more complex. \n")
print("Continuous refinement, diverse training data, and human oversight are crucial for developing truly responsible AI systems.")
print("________________________________________________________________\n")
