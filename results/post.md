---
title: The Villages That Went Quiet: what 2001 census data can and can't predict in Uttarakhand
published: true
tags: devchallenge, hf26challenge, opensource, ai
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge](https://dev.to/challenges/hf26) Week 4: Ghost in the Machine.*

## What I Built

Nothing in a ghost village is haunted. That is the unsettling part. There are no cold spots and no footsteps. People did not vanish: they left, one household at a time, for work, for a town with a clinic, for a life the village could not offer. The headlines count what is left behind, for example ["Migration empties 734 U'khand villages"](https://m.tribuneindia.com/article/migration-empties-734-u-khand-villages/696000). But if these places are haunted at all, they are haunted by a number: the one the census wrote down in 2001, and the smaller one it wrote down in 2011.

A census keeps a count the way an ofrenda keeps a name: so that someone, later, can look and say *they were here*.

So I did what a séance would do, with better paperwork. I took the 2001 count and asked it a question, using real data: **looking only at what the 2001 census recorded about a village (how many people lived there, what it had, who they were), can we tell which villages would have lost most of their people by 2011?**

What I built is a small, reproducible experiment, not a product:

- a table of **13,707 Uttarakhand villages** (6.2 million people in 2001), each described only by 2001 information;
- a label, fixed before any model was scored: **"depopulated" = 2011 population below 70% of the 2001 population, or no 2011 record at all**; 1,046 villages (7.6%) qualify;
- a fair test: **whole districts are held out**, so a model is always scored on districts it has never seen;
- three kinds of predictor compared against each other: a plain rule, standard models, and **TabPFN**, a pretrained model for small tables.

The short version of the result: with all 37 features, **TabPFN scored an AUC of 0.771** (95% interval 0.756 to 0.786), and 26.7% of the 10% of villages it ranked as most at risk really did lose people, against 7.6% for villages overall. The best standard model I tried, gradient boosting on the same features, scored 0.759 and 25.0%. That is a small edge with overlapping intervals, so I am **not** claiming TabPFN won. Neither model was tuned. The bigger finding is about the features: who lives in a village (age, work, literacy) told me more than what the village had (roads, clinics, schools).

There is no live demo. The deliverable is the experiment and the numbers. The tool never lists or ranks individual villages, on purpose, and the section near the end explains why.

## The quiet, in numbers

The first pattern is the plainest. The smaller the village, the more likely it went quiet:

| Village size in 2001 | Villages | Share that lost ≥30% of people (or vanished) |
|---|---|---|
| 50–99 | 2,145 | 16.3% |
| 100–249 | 4,941 | 9.4% |
| 250–499 | 3,567 | 3.6% |
| 500–999 | 1,884 | 2.5% |
| 1,000+ | 1,170 | 5.0% |

(The rise at 1,000+ is something I can't explain from this data. Large villages can be reclassified or merged between censuses, and I did not chase it.)

Amenities line up with it too. I counted four things a village could have in 2001: a paved road, a primary school, a medical facility and electricity.

| Of those four, a village had… | Villages | Share depopulated |
|---|---|---|
| none | 976 | 15.5% |
| one | 3,856 | 10.8% |
| two | 5,189 | 6.3% |
| three | 2,568 | 4.2% |
| all four | 1,118 | 3.8% |

**Read this carefully: it is an association, not a cause.** Small villages are also the ones that lack roads and schools, so I cannot say a road would have kept anyone. The model results below show how much the amenities add *after* size is accounted for. It is less than the table makes it look.

## How I built it

**Data.** [SHRUG](https://www.devdatalab.org/shrug), the open Socioeconomic High-resolution Rural-Urban Geographic Platform for India from the Development Data Lab, which links the same village across census years. I used its Population Census keys, the 2001 Population Census Abstract and the 2001 Village Directory for Uttarakhand (state code 05). 15,727 villages in 2001; 15,711 can be linked to 2011; 13,707 had at least 50 people in 2001 (below that, a drop from 18 to 12 people is noise, not migration).

**A catch in the data.** The 2011 table has no villages with zero people: the smallest village has 1 person. Villages that emptied completely are simply *missing* from it. I labelled units with no 2011 record as depopulated (64 of the 1,046), but some of those may be merges or renames, not exits. I cannot tell them apart, so the 64 are reported, not hidden.

**Features.** Two sets, written down before scoring.
- *SIZE* (6 features): log population, land area, population density, and the average size and density of the village's sub-district.
- *FULL* (37 features): SIZE plus 19 amenities from the Village Directory (schools, medical facilities, tap water, post office, bank, bus, paved and mud roads, distance to the nearest town, power, phone, irrigation, forest share) and 12 census shares (literacy, women's literacy, child share, caste and tribe shares, sex ratio, worker types, household size).

Every feature is from 2001. No 2011 number is ever an input.

**Evaluation.** Five folds, each holding out whole districts (13 districts in all). I report AUC, average precision, and the hit rate among the 10% of villages each model ranks as most at risk (the base rate is 7.6%), with bootstrap intervals.

**Models.**

| Model | Features | AUC (95% interval) | Hit rate, riskiest 10% (95% interval) |
|---|---|---|---|
| Plain rule: "smaller = riskier" | population only | 0.681 (0.663–0.698) | 17.2% (15.3–19.0) |
| Logistic regression | SIZE | 0.659 (0.640–0.677) | 14.8% (13.1–16.8) |
| Gradient boosting | SIZE | 0.702 (0.684–0.718) | 20.6% (18.5–22.8) |
| **TabPFN** | SIZE | **0.713** (0.697–0.729) | **19.1%** (16.9–21.3) |
| Logistic regression | FULL | 0.714 (0.698–0.730) | 21.5% (19.5–23.7) |
| Gradient boosting | FULL | 0.759 (0.743–0.773) | 25.0% (22.8–27.8) |
| **TabPFN** | FULL | **0.771** (0.756–0.786) | **26.7%** (24.2–29.0) |

![Dot-and-interval chart comparing AUC and top-10% hit rate for the plain rule, logistic regression, gradient boosting and TabPFN, with size-only and all-37-feature sets](⟦UPLOAD quiet_villages_results.png IN THE DEV EDITOR AND PASTE ITS LINK HERE⟧)

How to read it:

- With all 37 features, TabPFN is clearly ahead of the plain rule and of logistic regression. Against gradient boosting the gap is 0.012 AUC and the intervals overlap almost entirely. I did not run a paired test, so I call it a tie in practice.
- With size only, TabPFN (0.713) and gradient boosting (0.702) are also close, and gradient boosting is slightly ahead on hit rate (20.6% vs 19.1%).
- What is clear is the effect of the extra features. For TabPFN, going from 6 features to 37 raised AUC from 0.713 to 0.771 and the hit rate from 19.1% to 26.7%, and those intervals do not overlap.

**Which features mattered.** I asked it two ways. First an ablation with gradient boosting, same folds: village size alone gives an AUC of 0.702. Adding the 19 amenities lifts it to 0.721 (+0.019). Adding the 12 census shares instead lifts it to 0.749 (+0.047). Adding both gives 0.759. Second, permutation importance on the full model: shuffling village size cost the most AUC (0.078), then the share of workers in non-farm, non-household work (0.020), the child share (0.011) and women's literacy (0.009). The only amenity that registered was a primary school (0.005), then a paved road (0.003); power, banks and medical facilities were near zero once size was known. **Size and the shape of the population carry most of the signal; amenities add a little.** I would not read that as "roads don't matter": a 2001 census table cannot see whether a road was built after 2001, or whether people stayed because of one.

**TabPFN.** TabPFN is a pretrained model that classifies small tables directly, with no hand-tuning. I used it as-is. Its standard row limit is 10,000, so each training fold was down-sampled to 10,000 villages, stratified so the share of depopulated villages stayed the same. The same down-sample was used for both feature sets. The full run (5 folds, 2 feature sets, 10 fits) took about three minutes (173 seconds) on Colab.

## What I can't see from here

- **I started with a different project.** It was a scam-call screener. Before fine-tuning anything, I tested whether the untuned model already did the job. It scored 98% on 200 synthetic test calls, so a fine-tune had nothing real to improve, and the remaining errors came from my own synthetic data. I dropped it. The same habit ran through this project: fix the rules first, then look.
- **No 2021.** I wanted 2001 to 2021. India's 2021 census was never conducted; the next one is scheduled for 2027 ([2027 census of India](https://en.wikipedia.org/wiki/2027_census_of_India)). The data ends at 2011.
- **A mid-project change.** My first scoring used the SIZE features only, because the village directory had not arrived yet. When it arrived I added FULL, with its feature list fixed before I scored it. The label rules never changed. I had seen how many villages fall under the 70% line before choosing it, but not any model's results.
- **Only Uttarakhand, only rural villages, only 13 districts.** Nothing here says the same holds elsewhere.
- **Intervals are optimistic.** The bootstrap treats villages as independent, but villages in a district are not. Real uncertainty is wider.
- **Merged units.** SHRUG's village units are merged where census boundaries changed, so one unit can hide a single village emptying out.
- **Prediction is not explanation.** A model that ranks villages by risk does not say why people left. Work, land, roads and family all matter, and most of that is not in a census table.

## Why there is not one village name in this post

An ofrenda is made of names. This project has none: only village codes, counts and percentages. I thought about the obvious ending, a map of "the next villages to go quiet", and decided against it, for a reason I can put a number on.

Among the 10% of villages the best model ranked as most at risk, **26.7% really lost people. That means about three in four did not.** A ranked list from a model like this would be wrong about most of the villages on it. It would also hang the word "dying" on places where people live. A model can tell you that small villages tended to lose people between 2001 and 2011, and that those without a primary school or a paved road lost them somewhat more often (11.9% against 5.5%, and 8.7% against 4.8%, in the tables above). It cannot tell you which household is deciding tonight whether to go, and it should not be the one to write a village onto a list.

What the numbers can do is narrower and, I think, more useful: show that the quiet had a shape. It followed size first, then who lived there, and only a little what the village had. People who plan roads, clinics and schools can argue with that shape. I would rather leave them an argument than a hit list.

## Prize Categories

**Best Use of TabPFN.** TabPFN is the model under test, scored against the plain rule and standard models on the same held-out districts, with and without the amenity features.

I am not entering the other categories. I had planned a Tinker fine-tune for an earlier idea and dropped it before training anything, for the reason above, and I did not want to attach tools I did not really use.

## Why Open Innovation Matters

Everything here ran on public data. SHRUG is open under CC BY-NC-SA 4.0, and the whole experiment, from raw census files to the final table, can be rerun by anyone. TabPFN's weights are free for non-commercial use but need a license to be accepted first, so I would call it open to use, not open source in the strict sense. That distinction matters when the subject is real communities: anyone can check my numbers, and anyone can see exactly what the model was and was not given.

## Code and credits

The pipeline is four short Python scripts (build the table, baselines, ablation, TabPFN run). Derived data should be shared under CC BY-NC-SA 4.0, as SHRUG requires.

Data: Asher, Lunt, Matsuura and Novosad, *Development research at high geographic resolution: an analysis of night-lights, firms, and poverty in India using the SHRUG open data platform*, World Bank Economic Review 35(4), 2021. Census data from the Office of the Registrar General & Census Commissioner, India, via SHRUG 2.2.
