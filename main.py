from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

observed_A_value = 0  # 'fair'
observed_B_value = 0  # 'negative'
observed_C_value = 1  # 'not treated'
# Define the structure of the Bayesian Network
model = BayesianNetwork([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])

# Define conditional probability distributions (CPDs)
cpd_a = TabularCPD(variable='A', variable_card=3, values=[[0.7], [0.2], [0.1]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.8, 0.5, 0.3], [0.2, 0.5, 0.7]],
                   evidence=['A'], evidence_card=[3])
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.9, 0.7, 0.5, 0.3, 0.1, 0.5],
                           [0.1, 0.3, 0.5, 0.7, 0.9, 0.5]],
                   evidence=['A', 'B'], evidence_card=[3, 2])
cpd_d = TabularCPD(variable='D', variable_card=2,
                   values=[[0.8, 0.2],  
                           [0.2, 0.8]], 
                   evidence=['C'], evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)

# Check model validity
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# Make predictions
query_result = inference.query(variables=['C'], evidence={'A': 0, 'B': 0})
marginal_prob_A = inference.query(variables=['A'])

# Print the result
print("Probability distribution of Patient Condition A:")
print(marginal_prob_A)

# Perform inference
query_result_B = inference.query(variables=['B'], evidence={'A': 2})

# Print the result
print("Probability distribution of Test Result B given that patient's condition is poor:")
print(query_result_B)

# Perform inference
query_result_C = inference.query(variables=['C'], evidence={'A': 1, 'B': 1})

# Print the result
print("Probability distribution of Treatment Decision C given that patient's condition is fair and the test result is negative:")
print(query_result_C)

# Perform inference
query_result_D = inference.query(variables=['D'], evidence={'A': 1, 'B': 0, 'C': 1})

# Print the result
print("Probability distribution of Outcome D given that patient's condition is fair, the test result is positive, and the treatment decision is not treated:")
print(query_result_D)

# Perform MAP inference
most_likely_outcome = inference.map_query(variables=['D'], evidence={'A': observed_A_value, 'B': observed_B_value, 'C': observed_C_value})

# Print the most likely outcome
print("Most likely outcome D:", most_likely_outcome['D'])

# Perform MAP inference
predicted_treatment_decision = inference.map_query(variables=['C'], evidence={'A': observed_A_value, 'B': observed_B_value})

# Print the predicted treatment decision
print("Predicted Treatment Decision C:", predicted_treatment_decision['C'])

# Perform inference for both treatment decisions
query_result_treated = inference.query(variables=['D'], evidence={'C': 1})
query_result_not_treated = inference.query(variables=['D'], evidence={'C': 0})

# Get the probabilities for each treatment decision
prob_d_treated = query_result_treated.values
prob_d_not_treated = query_result_not_treated.values

# Calculate the sensitivity as the difference in probabilities
sensitivity = abs(prob_d_treated - prob_d_not_treated)

# Print the sensitivity
print("Sensitivity of outcome D to changes in treatment decision C:", sensitivity)  # Sensitivity of outcome D to changes in treatment decision C: [0.1 0.1]

# Define the combinations of values for variables A, B, and C
variable_combinations = [(a, b, c) for a in range(3) for b in range(2) for c in range(2)]

# Initialize a dictionary to store the probability distribution of outcome D for each combination
outcome_probabilities = {}

# Iterate over each combination of values for A, B, and C
for a, b, c in variable_combinations:
    # Perform inference to compute the probability distribution of outcome D for the current combination
    query_result = inference.query(variables=['D'], evidence={'A': a, 'B': b, 'C': c})
    outcome_probabilities[(a, b, c)] = query_result.values

# Print the probability distribution of outcome D for each combination
print("Probability distribution of patient's outcome D under different combinations of conditions:")
for combination, probability in outcome_probabilities.items():
    print(f"Combination (A={combination[0]}, B={combination[1]}, C={combination[2]}): {probability}")



# Get the probabilities for each treatment decision
treated_prob = query_result.values[0]
not_treated_prob = query_result.values[1]

# Display the result in a table
print("C\t\tD=Positive\t\tD=Negative")
print("treated\t\t {:.1f}\t\t\t{:.1f}".format(treated_prob, 1 - treated_prob))
print("not treated\t {:.1f}\t\t\t{:.1f}".format(not_treated_prob, 1 - not_treated_prob))
