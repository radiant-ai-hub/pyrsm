# Decision analysis tutorial (Python)

Python versions of the decision-tree materials of the [Radiant Tutorial Series](https://www.youtube.com/playlist?list=PLNhtaetb48EdKRIY7MewCyvb_1x7dV3xw), using `pyrsm.decide.dtree`. The trees are also available in Radiant under _Model > Decide > Decision analysis > Load example_ and in Python with `rsm.decide.dtree_examples()`.

| Notebook | Video | Example |
|---|---|---|
| [01-construct-a-tree](01-construct-a-tree.ipynb) | [#2 Using Radiant to Construct a Decision Tree](https://youtu.be/dBySH3z2Paw) | 1 |
| [02-trees-in-a-report](02-trees-in-a-report.ipynb) | [#3 How to Write Decision Tree Results into a Report](https://youtu.be/NlvWCU8UhTc) | 2 |
| [03-sensitivity](03-sensitivity.ipynb) | [#4 Sensitivity Analysis of Decision Tree](https://youtu.be/c_pCCCn6FEw) | 2 |
| [04-debug-tree-input](04-debug-tree-input.ipynb) | [#5 How to Debug Decision Tree Input](https://youtu.be/oiwv15bbjzs) | 2 |
| [05-imperfect-information](05-imperfect-information.ipynb) | [#6](https://youtu.be/xuv9zgAcvCQ) and [#7 Decision Trees with Imperfect Information](https://youtu.be/cVrYFFX3xN0) | 3, 4 |
| [06-subtrees](06-subtrees.ipynb) | [#8 Building a Decision Tree with Sub-trees](https://youtu.be/Mi-eZEoU9TE) | 2, 4 |

Start with [#1 Introduction to Decision Analysis](https://youtu.be/plSeVJ7c-Iw) (solving a tree by hand).

## Questions

**Example 1.** You are the owner of Radycal Kicks and deciding whether to open a new store. You've learned that there is a location available in a trendy district, but because of high rents it is a risky proposition. The fixed cost of opening the store is $2 million. Based on historical performance of your business and this district, you believe there is a 60% chance of the store being successful and earning $5 million going forward, and a 40% chance of it being a failure and losing $1 million going forward (these earnings and losses are in addition to the initial fixed cost of opening the store). What is the optimal strategy, and what is the Expected Monetary Value (EMV) associated with your strategy?

**Example 2.** You are an actor and would like to sign a contract with either a movie company or a TV network. If you sign with a movie company, you need to pay $5,000 for the legal fees. There will be a 30% chance of having small box office and earning $200,000, a 60% chance of having medium box office with payoff equal to $1,000,000, and a 10% chance of having large box office and getting $3,000,000. If you decide to sign the contract with TV networks instead, your payoff will be $900,000. What is EMV of signing the contract with movie company? What is your optimal strategy?

**Example 3.** You are the owner of Radycal Kicks and deciding whether to open a new store. You believe there is a 60% chance of the store being successful. Now you are offered to run a short-term popup store to better evaluate store performance in the district. Running a popup store helps identify whether or not the location will be successful. You believe that the popup test will correctly identify successful locations 80% of the time and correctly identify failures 90% of the time. Given the result of popup test is successful, what is the probability of a real store being successful?

**Example 4.** Continued from Example 1. Now the owner of the retail space offers you the ability to run a short-term popup store to better evaluate store performance in the district. Doing so will cost an additional $500,000 but allow you to acquire potentially helpful information. Running a popup store helps identify whether or not the location will be successful. You believe that the popup test will correctly identify successful locations 80% of the time and correctly identify failures 90% of the time. What is the optimal strategy? What is the EMV associated with your strategy?

## Answers

- **Open a new store (Example 1)**: Open the store: EMV $600,000 (vs. $0 for not opening).
- **Sign a contract (Example 2)**: Sign with the movie company: EMV $955,000 vs. $900,000 for the TV network. With $10,000 in legal fees the EMV is $950,000; with a $800,000 payoff for a medium box office it drops to $835,000 and signing with the TV network is best.
- **Sensitivity analysis (Example 2)**: Method 1 (edit the value): EMV $955,000 with $5,000 and $952,000 with $8,000 in legal fees. Method 2 (a variable): below $60,000 in legal fees sign with the movie company, above $60,000 with the TV network, indifferent at $60,000.
- **Debug decision tree input (Example 2)**: Four problems: line 4 'type: chance' is not indented under its node name, line 9 'p: 0.6' is not indented under 'Medium Box Office:', line 12 has no probability, and line 13 'payoff' has no ':' and no value. The corrected tree is the 'sign contract' tree of the Sign a contract example.
- **Imperfect information: a popup store (Examples 3 and 4)**: P(success | positive test) = 0.8 x 0.6 / (0.8 x 0.6 + 0.1 x 0.4) = 0.923. Run the popup test and open only after a positive result: EMV $820,000 vs. $600,000 for opening right away. 'parts' gives the same answer by referencing the 'popup' tree, which uses the variables of 'parts'.
- **Sub-trees (Example 2)**: The sub-tree (signing with the movie company) is worth $955,000; it uses the legal fees defined in 'main_tree'. Sign with the movie company: EMV $955,000 vs. $900,000.
