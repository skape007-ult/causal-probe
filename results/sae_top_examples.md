# SAE features: top-activating examples

Features were selected on nonsense stories 0-4 (AUROC for one graph type vs the rest) and are shown with their held-out AUROC on nonsense stories 5-9. 'My reading' is a human-style label written after looking at the examples; 'unclear' is used freely.


## Layer 6 · premise_end · structure = **confounding**


**Feature 7015** — AUROC discovery 0.64, held-out stories 0.65; fires on 90% of confounding vs 82% of others; top-8 purity 12%

- (diamondcut, nonsense8, ate, act 11.9) …x and are not pexu, the probability of rukz is 41%. For those who are hwax and are pexu, the probability of rukz is 61%. The overall probability of hwax is 96%. | Will pexu increase the chance of rukz?
- (diamondcut, nonsense8, ate, act 11.9) …x and are not pexu, the probability of rukz is 41%. For those who are hwax and are pexu, the probability of rukz is 61%. The overall probability of hwax is 96%. | Will pexu decrease the chance of rukz?
- (frontdoor, nonsense9, nie, act 11.7) …and are not jyka, the probability of glimx is 30%. For those who are zuph and are jyka, the probability of glimx is 38%. The overall probability of zuph is 38%. | Does zuph negatively affect glimx through jyka?
- (diamond, nonsense8, det-counterfactual, act 11.5) …fect on kraz and hwax. Hwax has a direct effect on rukz. Kraz has a direct effect on rukz. We know that pexu causes hwax and not kraz. hwax or kraz causes rukz. | Would an individual is not rukz if not pexu instead of pexu?
- (frontdoor, nonsense8, nie, act 11.4) …xu and are not hwax, the probability of rukz is 8%. For those who are pexu and are hwax, the probability of rukz is 82%. The overall probability of pexu is 82%. | Does pexu positively affect rukz through hwax?

**Feature 11631** — AUROC discovery 0.63, held-out stories 0.64; fires on 49% of confounding vs 21% of others; top-8 purity 0%

- (frontdoor, nonsense5, det-counterfactual, act 3.7) …s a direct effect on kwox. Qwiu is unobserved. We know that qwiu causes jyka. jyka causes not yupt. qwiu or yupt causes kwox. We observed an individual is qwiu. | Would an individual is not kwox if jyka instead of not jyka?
- (frontdoor, nonsense5, det-counterfactual, act 3.6) …t has a direct effect on kwox. Qwiu is unobserved. We know that qwiu causes jyka. jyka causes yupt. qwiu or yupt causes kwox. We observed an individual is qwiu. | Would an individual is not kwox if not jyka instead of jyka?
- (frontdoor, nonsense7, det-counterfactual, act 3.6) … a direct effect on uvzi. Vubr is unobserved. We know that vubr causes zuph. zuph causes not wibl. vubr and wibl causes uvzi. We observed an individual is vubr. | Would an individual is not uvzi if not zuph instead of zuph?
- (diamondcut, nonsense8, ate, act 3.6) …x and are not pexu, the probability of rukz is 76%. For those who are hwax and are pexu, the probability of rukz is 22%. The overall probability of hwax is 43%. | Will pexu increase the chance of rukz?
- (frontdoor, nonsense6, det-counterfactual, act 3.6) …s a direct effect on muvq. Xyfo is unobserved. We know that xyfo causes yupt. yupt causes not zupj. xyfo or zupj causes muvq. We observed an individual is xyfo. | Would an individual is muvq if not yupt instead of yupt?

**Feature 14447** — AUROC discovery 0.63, held-out stories 0.64; fires on 56% of confounding vs 36% of others; top-8 purity 25%

- (diamondcut, nonsense3, ate, act 6.0) …t and are not yomx, the probability of xevu is 72%. For those who are gwet and are yomx, the probability of xevu is 88%. The overall probability of gwet is 12%. | Will yomx increase the chance of xevu?
- (confounding, nonsense8, ett, act 5.9) …ax and are not pexu, the probability of rukz is 58%. For those who are hwax and are pexu, the probability of rukz is 56%. The overall probability of hwax is 1%. | For those who are pexu, would it be more likely to see rukz if the individual was not pexu?
- (confounding, nonsense3, ate, act 5.6) …t and are not yomx, the probability of xevu is 36%. For those who are gwet and are yomx, the probability of xevu is 63%. The overall probability of gwet is 12%. | Will yomx increase the chance of xevu?
- (diamondcut, nonsense5, correlation, act 5.6) …. Qwiu has a direct effect on kwox. The overall probability of jyka is 15%. The probability of not jyka and kwox is 16%. The probability of jyka and kwox is 2%. | Is the chance of kwox smaller when observing jyka?
- (diamondcut, nonsense2, ate, act 5.4) …v and are not xevo, the probability of gyzp is 92%. For those who are tijv and are xevo, the probability of gyzp is 73%. The overall probability of tijv is 13%. | Will xevo increase the chance of gyzp?

**Feature 3956** — AUROC discovery 0.63, held-out stories 0.59; fires on 94% of confounding vs 82% of others; top-8 purity 0%

- (diamondcut, nonsense8, backadj, act 10.2) …ct effect on rukz. Method 1: We look directly at how pexu correlates with rukz in general. Method 2: We look at this correlation case by case according to kraz. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense8, backadj, act 10.2) …ct effect on rukz. Method 1: We look directly at how pexu correlates with rukz in general. Method 2: We look at this correlation case by case according to kraz. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense8, backadj, act 10.2) …ct effect on rukz. Method 1: We look directly at how pexu correlates with rukz in general. Method 2: We look at this correlation case by case according to kraz. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense8, backadj, act 10.2) …ct effect on rukz. Method 1: We look directly at how pexu correlates with rukz in general. Method 2: We look at this correlation case by case according to kraz. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense7, backadj, act 10.1) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?

**Feature 10359** — AUROC discovery 0.60, held-out stories 0.60; fires on 47% of confounding vs 29% of others; top-8 purity 0%

- (mediation, nonsense2, ate, act 7.3) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?
- (frontdoor, nonsense9, nie, act 6.7) …and are not jyka, the probability of glimx is 30%. For those who are zuph and are jyka, the probability of glimx is 38%. The overall probability of zuph is 38%. | Does zuph negatively affect glimx through jyka?
- (frontdoor, nonsense1, nie, act 6.1) …x and are not swoq, the probability of kwoz is 67%. For those who are kwox and are swoq, the probability of kwoz is 53%. The overall probability of kwox is 53%. | Does kwox negatively affect kwoz through swoq?
- (diamondcut, nonsense2, ate, act 6.0) …v and are not xevo, the probability of gyzp is 92%. For those who are tijv and are xevo, the probability of gyzp is 73%. The overall probability of tijv is 13%. | Will xevo increase the chance of gyzp?
- (diamondcut, nonsense0, ate, act 5.9) …h and are not rixq, the probability of xevu is 56%. For those who are zuph and are rixq, the probability of xevu is 61%. The overall probability of zuph is 53%. | Will rixq increase the chance of xevu?

## Layer 6 · premise_end · structure = **collision**


**Feature 2672** — AUROC discovery 0.73, held-out stories 0.62; fires on 50% of collision vs 19% of others; top-8 purity 0%

- (fork, nonsense7, backadj, act 4.5) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense7, backadj, act 4.5) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?
- (IV, nonsense2, backadj, act 4.3) …ijw is unobserved. Method 1: We look directly at how xevo correlates with gyzp in general. Method 2: We look at this correlation case by case according to tijw. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (frontdoor, nonsense2, backadj, act 4.1) …ijw is unobserved. Method 1: We look directly at how xevo correlates with gyzp in general. Method 2: We look at this correlation case by case according to tijw. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense7, nie, act 4.0) …t on wibl. Wibl has a direct effect on uvzi. For those who are not zuph, the probability of uvzi is 97%. For those who are zuph, the probability of uvzi is 97%. | Does zuph positively affect uvzi through wibl?

**Feature 2193** — AUROC discovery 0.73, held-out stories 0.61; fires on 56% of collision vs 29% of others; top-8 purity 25%

- (arrowhead, nonsense5, correlation, act 7.0) … effect on kwox. Qwiu is unobserved. The overall probability of jyka is 0%. The probability of not jyka and kwox is 11%. The probability of jyka and kwox is 0%. | Is the chance of kwox smaller when observing jyka?
- (fork, nonsense8, correlation, act 6.2) …z. Hwax has a direct effect on rukz. The overall probability of pexu is 0%. The probability of not pexu and rukz is 44%. The probability of pexu and rukz is 0%. | Is the chance of rukz smaller when observing pexu?
- (collision, nonsense8, collider_bias, act 5.3) …ausal relationships: Pexu has a direct effect on hwax. Rukz has a direct effect on hwax. For those who are hwax, the correlation between pexu and rukz is -0.00. | If we look at those who are hwax, does it mean that pexu affects rukz?
- (arrowhead, nonsense6, nie, act 5.1) …pt and are not xyfo, the probability of zupj is 49%. For those who are yupt and are xyfo, the probability of zupj is 8%. The overall probability of xyfo is 28%. | Does yupt negatively affect muvq through zupj?
- (diamondcut, nonsense6, correlation, act 4.9) …q. Xyfo has a direct effect on muvq. The overall probability of yupt is 8%. The probability of not yupt and muvq is 85%. The probability of yupt and muvq is 4%. | Is the chance of muvq smaller when observing yupt?

**Feature 7002** — AUROC discovery 0.72, held-out stories 0.69; fires on 100% of collision vs 100% of others; top-8 purity 25%

- (mediation, nonsense2, backadj, act 32.2) …gyzp. Method 1: We look at how xevo correlates with gyzp case by case according to tijv. Method 2: We look directly at how xevo correlates with gyzp in general. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense2, backadj, act 32.0) …gyzp. Method 1: We look at how xevo correlates with gyzp case by case according to tijv. Method 2: We look directly at how xevo correlates with gyzp in general. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense2, backadj, act 32.0) …gyzp. Method 1: We look at how xevo correlates with gyzp case by case according to tijv. Method 2: We look directly at how xevo correlates with gyzp in general. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense2, backadj, act 32.0) …gyzp. Method 1: We look at how xevo correlates with gyzp case by case according to tijv. Method 2: We look directly at how xevo correlates with gyzp in general. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense2, backadj, act 32.0) …gyzp. Method 1: We look at how xevo correlates with gyzp case by case according to tijv. Method 2: We look directly at how xevo correlates with gyzp in general. | To understand how xevo affects gyzp, is it more correct to use the Method 1 than Method 2?

**Feature 8000** — AUROC discovery 0.71, held-out stories 0.64; fires on 61% of collision vs 31% of others; top-8 purity 25%

- (collision, nonsense7, backadj, act 7.1) …ct effect on wibl. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense6, det-counterfactual, act 6.4) …sal relationships: Yupt has a direct effect on muvq. Zupj has a direct effect on muvq. We know that yupt or zupj causes muvq. We observed an individual is zupj. | Would an individual is muvq if yupt instead of not yupt?
- (fork, nonsense6, det-counterfactual, act 6.4) …sal relationships: Yupt has a direct effect on muvq. Zupj has a direct effect on muvq. We know that yupt or zupj causes muvq. We observed an individual is zupj. | Would an individual is not muvq if not yupt instead of yupt?
- (chain, nonsense7, backadj, act 6.3) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?
- (fork, nonsense7, backadj, act 5.9) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?

**Feature 3742** — AUROC discovery 0.70, held-out stories 0.75; fires on 61% of collision vs 16% of others; top-8 purity 0%

- (frontdoor, nonsense5, backadj, act 6.2) …rved. Method 1: We look at how jyka correlates with kwox case by case according to qwiu. Method 2: We look directly at how jyka correlates with kwox in general. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (frontdoor, nonsense6, backadj, act 6.2) …rved. Method 1: We look at how yupt correlates with muvq case by case according to xyfo. Method 2: We look directly at how yupt correlates with muvq in general. | To understand how yupt affects muvq, is it more correct to use the Method 1 than Method 2?
- (frontdoor, nonsense6, backadj, act 6.2) …rved. Method 1: We look at how yupt correlates with muvq case by case according to xyfo. Method 2: We look directly at how yupt correlates with muvq in general. | To understand how yupt affects muvq, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense5, backadj, act 6.1) …kwox. Method 1: We look at how jyka correlates with kwox case by case according to qwiu. Method 2: We look directly at how jyka correlates with kwox in general. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense5, backadj, act 6.1) …kwox. Method 1: We look at how jyka correlates with kwox case by case according to qwiu. Method 2: We look directly at how jyka correlates with kwox in general. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?

## Layer 6 · premise_end · structure = **chain**


**Feature 13460** — AUROC discovery 0.79, held-out stories 0.84; fires on 100% of chain vs 99% of others; top-8 purity 62%

- (chain, nonsense5, backadj, act 7.7) …ct effect on kwox. Method 1: We look directly at how jyka correlates with kwox in general. Method 2: We look at this correlation case by case according to yupt. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense5, backadj, act 7.7) …ct effect on kwox. Method 1: We look directly at how jyka correlates with kwox in general. Method 2: We look at this correlation case by case according to yupt. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense3, backadj, act 7.7) …ct effect on gwet. Method 1: We look directly at how yomx correlates with xevu in general. Method 2: We look at this correlation case by case according to gwet. | To understand how yomx affects xevu, is it more correct to use the Method 1 than Method 2?
- (confounding, nonsense4, backadj, act 7.7) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to hwax. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (confounding, nonsense4, backadj, act 7.7) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to hwax. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?

**Feature 2094** — AUROC discovery 0.73, held-out stories 0.72; fires on 87% of chain vs 42% of others; top-8 purity 0%

- (confounding, nonsense1, correlation, act 4.6) … Kwox has a direct effect on kwoz. The overall probability of kwox is 65%. The probability of not kwox and kwoz is 16%. The probability of kwox and kwoz is 50%. | Is the chance of kwoz larger when observing kwox?
- (diamond, nonsense6, ate, act 4.3) …t on muvq. Xyfo has a direct effect on muvq. For those who are not yupt, the probability of muvq is 60%. For those who are yupt, the probability of muvq is 55%. | Will yupt decrease the chance of muvq?
- (arrowhead, nonsense6, correlation, act 4.3) …effect on muvq. Xyfo is unobserved. The overall probability of yupt is 90%. The probability of not yupt and muvq is 7%. The probability of yupt and muvq is 50%. | Is the chance of muvq smaller when observing yupt?
- (confounding, nonsense8, correlation, act 4.2) … Pexu has a direct effect on rukz. The overall probability of pexu is 67%. The probability of not pexu and rukz is 15%. The probability of pexu and rukz is 54%. | Is the chance of rukz smaller when observing pexu?
- (diamondcut, nonsense8, correlation, act 4.2) …. Kraz has a direct effect on rukz. The overall probability of pexu is 58%. The probability of not pexu and rukz is 8%. The probability of pexu and rukz is 50%. | Is the chance of rukz larger when observing pexu?

**Feature 6058** — AUROC discovery 0.70, held-out stories 0.73; fires on 85% of chain vs 63% of others; top-8 purity 0%

- (mediation, nonsense2, ate, act 5.5) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?
- (diamond, nonsense2, nie, act 5.3) …t on gyzp. Tijw has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 80%. For those who are xevo, the probability of gyzp is 36%. | Does xevo positively affect gyzp through tijv and tijw?
- (fork, nonsense4, ate, act 5.3) …t on lirg. Hwax has a direct effect on lirg. For those who are not jyka, the probability of lirg is 71%. For those who are jyka, the probability of lirg is 36%. | Will jyka decrease the chance of lirg?
- (fork, nonsense6, ate, act 5.2) …t on muvq. Zupj has a direct effect on muvq. For those who are not yupt, the probability of muvq is 89%. For those who are yupt, the probability of muvq is 22%. | Will yupt decrease the chance of muvq?
- (fork, nonsense2, ate, act 5.2) …t on gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 82%. For those who are xevo, the probability of gyzp is 30%. | Will xevo increase the chance of gyzp?

**Feature 8204** — AUROC discovery 0.69, held-out stories 0.69; fires on 72% of chain vs 44% of others; top-8 purity 50%

- (diamondcut, nonsense9, ett, act 6.8) …d are zory, the probability of glimx is 83%. For those who are not zuph, the probability of zory is 97%. For those who are zuph, the probability of zory is 97%. | For those who are zuph, would it be less likely to see glimx if the individual was not zuph?
- (chain, nonsense5, nie, act 6.7) …t on yupt. Yupt has a direct effect on kwox. For those who are not jyka, the probability of kwox is 71%. For those who are jyka, the probability of kwox is 67%. | Does jyka negatively affect kwox through yupt?
- (chain, nonsense5, nie, act 6.7) …t on yupt. Yupt has a direct effect on kwox. For those who are not jyka, the probability of kwox is 71%. For those who are jyka, the probability of kwox is 67%. | Does jyka positively affect kwox through yupt?
- (fork, nonsense4, ate, act 6.6) …t on lirg. Hwax has a direct effect on lirg. For those who are not jyka, the probability of lirg is 20%. For those who are jyka, the probability of lirg is 40%. | Will jyka decrease the chance of lirg?
- (diamond, nonsense2, ett, act 6.5) …t on gyzp. Tijw has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 47%. For those who are xevo, the probability of gyzp is 49%. | For those who are xevo, would it be less likely to see gyzp if the individual was not xevo?

**Feature 10961** — AUROC discovery 0.67, held-out stories 0.68; fires on 99% of chain vs 88% of others; top-8 purity 62%

- (chain, nonsense3, det-counterfactual, act 9.5) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (chain, nonsense6, det-counterfactual, act 9.4) …s or causal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes zupj, and we know that zupj causes muvq. | Would an individual is not muvq if yupt instead of not yupt?
- (chain, nonsense6, det-counterfactual, act 9.3) …sal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes not zupj, and we know that zupj causes not muvq. | Would an individual is not muvq if not yupt instead of yupt?
- (chain, nonsense0, det-counterfactual, act 9.3) …s or causal relationships: Rixq has a direct effect on zuph. Zuph has a direct effect on xevu. We know that rixq causes zuph, and we know that zuph causes xevu. | Would an individual is xevu if not rixq instead of rixq?
- (mediation, nonsense1, det-counterfactual, act 9.3) … causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes not swoq. kwox or swoq causes kwoz. | Would an individual is not kwoz if not kwox instead of kwox?

## Layer 6 · premise_end · structure = **fork**


**Feature 13460** — AUROC discovery 0.82, held-out stories 0.78; fires on 100% of fork vs 99% of others; top-8 purity 0%

- (chain, nonsense5, backadj, act 7.7) …ct effect on kwox. Method 1: We look directly at how jyka correlates with kwox in general. Method 2: We look at this correlation case by case according to yupt. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense5, backadj, act 7.7) …ct effect on kwox. Method 1: We look directly at how jyka correlates with kwox in general. Method 2: We look at this correlation case by case according to yupt. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense3, backadj, act 7.7) …ct effect on gwet. Method 1: We look directly at how yomx correlates with xevu in general. Method 2: We look at this correlation case by case according to gwet. | To understand how yomx affects xevu, is it more correct to use the Method 1 than Method 2?
- (confounding, nonsense4, backadj, act 7.7) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to hwax. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (confounding, nonsense4, backadj, act 7.7) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to hwax. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?

**Feature 6035** — AUROC discovery 0.76, held-out stories 0.66; fires on 71% of fork vs 32% of others; top-8 purity 62%

- (fork, nonsense8, det-counterfactual, act 6.9) …elationships: Pexu has a direct effect on rukz. Hwax has a direct effect on rukz. We know that pexu and hwax causes rukz. We observed an individual is not hwax. | Would an individual is not rukz if not pexu instead of pexu?
- (fork, nonsense8, det-counterfactual, act 6.9) …elationships: Pexu has a direct effect on rukz. Hwax has a direct effect on rukz. We know that pexu and hwax causes rukz. We observed an individual is not hwax. | Would an individual is rukz if pexu instead of not pexu?
- (diamondcut, nonsense8, det-counterfactual, act 6.8) …t effect on rukz. Kraz has a direct effect on rukz. We know that hwax causes not pexu and kraz. pexu or kraz causes rukz. We observed an individual is not hwax. | Would an individual is not rukz if not pexu instead of pexu?
- (fork, nonsense3, det-counterfactual, act 6.4) …elationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is not gwet. | Would an individual is not xevu if not yomx instead of yomx?
- (fork, nonsense3, det-counterfactual, act 6.4) …elationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is not gwet. | Would an individual is xevu if not yomx instead of yomx?

**Feature 13216** — AUROC discovery 0.70, held-out stories 0.59; fires on 45% of fork vs 16% of others; top-8 purity 38%

- (mediation, nonsense6, ett, act 9.6) … and muvq. Zupj has a direct effect on muvq. For those who are not yupt, the probability of muvq is 73%. For those who are yupt, the probability of muvq is 18%. | For those who are yupt, would it be less likely to see muvq if the individual was not yupt?
- (mediation, nonsense5, ate, act 8.7) … and kwox. Yupt has a direct effect on kwox. For those who are not jyka, the probability of kwox is 81%. For those who are jyka, the probability of kwox is 27%. | Will jyka decrease the chance of kwox?
- (arrowhead, nonsense7, ate, act 8.5) …a direct effect on uvzi. Vubr is unobserved. For those who are not zuph, the probability of uvzi is 65%. For those who are zuph, the probability of uvzi is 18%. | Will zuph decrease the chance of uvzi?
- (mediation, nonsense4, ett, act 8.4) … and lirg. Hwax has a direct effect on lirg. For those who are not jyka, the probability of lirg is 67%. For those who are jyka, the probability of lirg is 18%. | For those who are jyka, would it be less likely to see lirg if the individual was not jyka?
- (fork, nonsense2, ett, act 8.2) …t on gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 77%. For those who are xevo, the probability of gyzp is 22%. | For those who are xevo, would it be more likely to see gyzp if the individual was not xevo?

**Feature 6058** — AUROC discovery 0.69, held-out stories 0.70; fires on 85% of fork vs 64% of others; top-8 purity 62%

- (mediation, nonsense2, ate, act 5.5) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?
- (diamond, nonsense2, nie, act 5.3) …t on gyzp. Tijw has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 80%. For those who are xevo, the probability of gyzp is 36%. | Does xevo positively affect gyzp through tijv and tijw?
- (fork, nonsense4, ate, act 5.3) …t on lirg. Hwax has a direct effect on lirg. For those who are not jyka, the probability of lirg is 71%. For those who are jyka, the probability of lirg is 36%. | Will jyka decrease the chance of lirg?
- (fork, nonsense6, ate, act 5.2) …t on muvq. Zupj has a direct effect on muvq. For those who are not yupt, the probability of muvq is 89%. For those who are yupt, the probability of muvq is 22%. | Will yupt decrease the chance of muvq?
- (fork, nonsense2, ate, act 5.2) …t on gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 82%. For those who are xevo, the probability of gyzp is 30%. | Will xevo increase the chance of gyzp?

**Feature 2533** — AUROC discovery 0.64, held-out stories 0.62; fires on 80% of fork vs 73% of others; top-8 purity 25%

- (chain, nonsense5, ett, act 7.8) …ect on yupt. Yupt has a direct effect on kwox. For those who are not jyka, the probability of kwox is 6%. For those who are jyka, the probability of kwox is 8%. | For those who are jyka, would it be more likely to see kwox if the individual was not jyka?
- (diamond, nonsense1, ett, act 7.7) …t on kwoz. Muvy has a direct effect on kwoz. For those who are not kwox, the probability of kwoz is 34%. For those who are kwox, the probability of kwoz is 40%. | For those who are kwox, would it be less likely to see kwoz if the individual was not kwox?
- (fork, nonsense2, ate, act 7.6) …t on gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 82%. For those who are xevo, the probability of gyzp is 30%. | Will xevo increase the chance of gyzp?
- (confounding, nonsense4, marginal, act 7.6) …lirg. The overall probability of jyka is 17%. For those who are not jyka, the probability of lirg is 46%. For those who are jyka, the probability of lirg is 8%. | Is lirg more likely than not lirg overall?
- (mediation, nonsense4, ett, act 7.4) … and lirg. Hwax has a direct effect on lirg. For those who are not jyka, the probability of lirg is 67%. For those who are jyka, the probability of lirg is 18%. | For those who are jyka, would it be less likely to see lirg if the individual was not jyka?

## Layer 6 · premise_end · structure = **mediation**


**Feature 7426** — AUROC discovery 0.68, held-out stories 0.58; fires on 46% of mediation vs 24% of others; top-8 purity 12%

- (arrowhead, nonsense6, correlation, act 4.2) …effect on muvq. Xyfo is unobserved. The overall probability of yupt is 90%. The probability of not yupt and muvq is 7%. The probability of yupt and muvq is 50%. | Is the chance of muvq smaller when observing yupt?
- (frontdoor, nonsense5, ett, act 4.2) …nd are yupt, the probability of kwox is 28%. For those who are not jyka, the probability of yupt is 93%. For those who are jyka, the probability of yupt is 25%. | For those who are jyka, would it be less likely to see kwox if the individual was not jyka?
- (diamondcut, nonsense0, correlation, act 4.1) … Swoy has a direct effect on xevu. The overall probability of rixq is 50%. The probability of not rixq and xevu is 49%. The probability of rixq and xevu is 44%. | Is the chance of xevu smaller when observing rixq?
- (frontdoor, nonsense0, ett, act 4.0) …nd are zuph, the probability of xevu is 46%. For those who are not rixq, the probability of zuph is 76%. For those who are rixq, the probability of zuph is 55%. | For those who are rixq, would it be less likely to see xevu if the individual was not rixq?
- (mediation, nonsense6, nde, act 3.9) …and are zupj, the probability of muvq is 4%. For those who are not yupt, the probability of zupj is 56%. For those who are yupt, the probability of zupj is 53%. | If we disregard the mediation effect through zupj, would yupt negatively affect muvq?

**Feature 6201** — AUROC discovery 0.66, held-out stories 0.54; fires on 76% of mediation vs 59% of others; top-8 purity 12%

- (collision, nonsense1, exp_away, act 13.5) …ity of kwoz is 96%. For those who are kwox and are not swoq, the probability of kwoz is 74%. For those who are kwox and are swoq, the probability of kwoz is 6%. | If we look at those who are swoq, does the chance of kwoz increase when kwox?
- (diamondcut, nonsense5, ett, act 13.3) … and are qwiu, the probability of kwox is 94%. For those who are not jyka, the probability of qwiu is 1%. For those who are jyka, the probability of qwiu is 5%. | For those who are jyka, would it be more likely to see kwox if the individual was not jyka?
- (diamondcut, nonsense0, ett, act 12.5) …and are swoy, the probability of xevu is 44%. For those who are not rixq, the probability of swoy is 15%. For those who are rixq, the probability of swoy is 9%. | For those who are rixq, would it be less likely to see xevu if the individual was not rixq?
- (frontdoor, nonsense7, ett, act 11.7) …nd are wibl, the probability of uvzi is 78%. For those who are not zuph, the probability of wibl is 93%. For those who are zuph, the probability of wibl is 15%. | For those who are zuph, would it be less likely to see uvzi if the individual was not zuph?
- (diamondcut, nonsense1, ett, act 11.2) … and are muvy, the probability of kwoz is 92%. For those who are not kwox, the probability of muvy is 9%. For those who are kwox, the probability of muvy is 8%. | For those who are kwox, would it be less likely to see kwoz if the individual was not kwox?

**Feature 8512** — AUROC discovery 0.65, held-out stories 0.55; fires on 82% of mediation vs 76% of others; top-8 purity 50%

- (collision, nonsense0, collider_bias, act 6.2) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.01. | If we look at those who are zuph, does it mean that rixq affects xevu?
- (collision, nonsense4, collider_bias, act 6.1) …causal relationships: Jyka has a direct effect on hwax. Lirg has a direct effect on hwax. For those who are hwax, the correlation between jyka and lirg is 0.17. | If we look at those who are hwax, does it mean that jyka does not affect lirg?
- (collision, nonsense0, collider_bias, act 6.0) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.07. | If we look at those who are zuph, does it mean that rixq does not affect xevu?
- (diamond, nonsense0, det-counterfactual, act 6.0) … effect on swoy and zuph. Zuph has a direct effect on xevu. Swoy has a direct effect on xevu. We know that rixq causes zuph and swoy. zuph and swoy causes xevu. | Would an individual is xevu if rixq instead of not rixq?
- (mediation, nonsense0, det-counterfactual, act 6.0) … or causal relationships: Rixq has a direct effect on zuph and xevu. Zuph has a direct effect on xevu. We know that rixq causes zuph. rixq and zuph causes xevu. | Would an individual is xevu if not rixq instead of rixq?

**Feature 9786** — AUROC discovery 0.64, held-out stories 0.56; fires on 23% of mediation vs 5% of others; top-8 purity 25%

- (mediation, nonsense2, ate, act 5.3) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?
- (diamondcut, nonsense0, correlation, act 4.2) …. Swoy has a direct effect on xevu. The overall probability of rixq is 17%. The probability of not rixq and xevu is 65%. The probability of rixq and xevu is 6%. | Is the chance of xevu smaller when observing rixq?
- (mediation, nonsense3, correlation, act 3.6) …. Gwet has a direct effect on xevu. The overall probability of yomx is 65%. The probability of not yomx and xevu is 11%. The probability of yomx and xevu is 6%. | Is the chance of xevu larger when observing yomx?
- (chain, nonsense6, det-counterfactual, act 3.6) …sal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes not zupj, and we know that zupj causes not muvq. | Would an individual is not muvq if not yupt instead of yupt?
- (chain, nonsense8, det-counterfactual, act 3.3) …sal relationships: Pexu has a direct effect on hwax. Hwax has a direct effect on rukz. We know that pexu causes not hwax, and we know that hwax causes not rukz. | Would an individual is not rukz if pexu instead of not pexu?

**Feature 10007** — AUROC discovery 0.64, held-out stories 0.57; fires on 23% of mediation vs 5% of others; top-8 purity 38%

- (diamond, nonsense7, det-counterfactual, act 3.5) …t effect on vubr and wibl. Wibl has a direct effect on uvzi. Vubr has a direct effect on uvzi. We know that zuph causes wibl and vubr. wibl or vubr causes uvzi. | Would an individual is uvzi if zuph instead of not zuph?
- (diamond, nonsense7, det-counterfactual, act 3.5) …t effect on vubr and wibl. Wibl has a direct effect on uvzi. Vubr has a direct effect on uvzi. We know that zuph causes wibl and vubr. wibl or vubr causes uvzi. | Would an individual is uvzi if not zuph instead of zuph?
- (diamond, nonsense7, det-counterfactual, act 3.5) …t effect on vubr and wibl. Wibl has a direct effect on uvzi. Vubr has a direct effect on uvzi. We know that zuph causes wibl and vubr. wibl or vubr causes uvzi. | Would an individual is not uvzi if zuph instead of not zuph?
- (mediation, nonsense7, det-counterfactual, act 3.2) … or causal relationships: Zuph has a direct effect on wibl and uvzi. Wibl has a direct effect on uvzi. We know that zuph causes wibl. zuph and wibl causes uvzi. | Would an individual is uvzi if zuph instead of not zuph?
- (mediation, nonsense6, det-counterfactual, act 3.1) … or causal relationships: Yupt has a direct effect on zupj and muvq. Zupj has a direct effect on muvq. We know that yupt causes zupj. yupt and zupj causes muvq. | Would an individual is muvq if not yupt instead of yupt?

## Layer 12 · premise_end · structure = **confounding**


**Feature 7507** — AUROC discovery 0.68, held-out stories 0.62; fires on 59% of confounding vs 29% of others; top-8 purity 12%

- (arrowhead, nonsense5, nie, act 22.9) …ka and are not qwiu, the probability of yupt is 82%. For those who are jyka and are qwiu, the probability of yupt is 45%. The overall probability of qwiu is 1%. | Does jyka positively affect kwox through yupt?
- (fork, nonsense0, det-counterfactual, act 21.1) …al relationships: Rixq has a direct effect on xevu. Zuph has a direct effect on xevu. We know that rixq and zuph causes xevu. We observed an individual is zuph. | Would an individual is not xevu if not rixq instead of rixq?
- (confounding, nonsense8, ett, act 20.3) …ax and are not pexu, the probability of rukz is 58%. For those who are hwax and are pexu, the probability of rukz is 56%. The overall probability of hwax is 1%. | For those who are pexu, would it be more likely to see rukz if the individual was not pexu?
- (fork, nonsense3, det-counterfactual, act 20.1) …al relationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is gwet. | Would an individual is not xevu if yomx instead of not yomx?
- (fork, nonsense3, det-counterfactual, act 20.1) …al relationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is gwet. | Would an individual is not xevu if not yomx instead of yomx?

**Feature 16346** — AUROC discovery 0.67, held-out stories 0.68; fires on 99% of confounding vs 95% of others; top-8 purity 0%

- (arrowhead, nonsense2, nie, act 19.7) …o and are not tijw, the probability of tijv is 18%. For those who are xevo and are tijw, the probability of tijv is 91%. The overall probability of tijw is 60%. | Does xevo negatively affect gyzp through tijv?
- (arrowhead, nonsense2, nie, act 19.7) …o and are not tijw, the probability of tijv is 18%. For those who are xevo and are tijw, the probability of tijv is 91%. The overall probability of tijw is 60%. | Does xevo positively affect gyzp through tijv?
- (arrowhead, nonsense2, nie, act 19.5) …o and are not tijw, the probability of tijv is 65%. For those who are xevo and are tijw, the probability of tijv is 41%. The overall probability of tijw is 28%. | Does xevo positively affect gyzp through tijv?
- (arrowhead, nonsense5, nie, act 19.3) …a and are not qwiu, the probability of yupt is 40%. For those who are jyka and are qwiu, the probability of yupt is 10%. The overall probability of qwiu is 66%. | Does jyka negatively affect kwox through yupt?
- (arrowhead, nonsense6, nie, act 18.9) …pt and are not xyfo, the probability of zupj is 49%. For those who are yupt and are xyfo, the probability of zupj is 8%. The overall probability of xyfo is 28%. | Does yupt negatively affect muvq through zupj?

**Feature 263** — AUROC discovery 0.67, held-out stories 0.68; fires on 64% of confounding vs 34% of others; top-8 purity 0%

- (fork, nonsense0, det-counterfactual, act 11.6) …sal relationships: Rixq has a direct effect on xevu. Zuph has a direct effect on xevu. We know that rixq or zuph causes xevu. We observed an individual is zuph. | Would an individual is not xevu if rixq instead of not rixq?
- (fork, nonsense0, det-counterfactual, act 11.6) …sal relationships: Rixq has a direct effect on xevu. Zuph has a direct effect on xevu. We know that rixq or zuph causes xevu. We observed an individual is zuph. | Would an individual is xevu if rixq instead of not rixq?
- (fork, nonsense6, det-counterfactual, act 11.3) …sal relationships: Yupt has a direct effect on muvq. Zupj has a direct effect on muvq. We know that yupt or zupj causes muvq. We observed an individual is zupj. | Would an individual is muvq if yupt instead of not yupt?
- (fork, nonsense6, det-counterfactual, act 11.2) …sal relationships: Yupt has a direct effect on muvq. Zupj has a direct effect on muvq. We know that yupt or zupj causes muvq. We observed an individual is zupj. | Would an individual is not muvq if not yupt instead of yupt?
- (fork, nonsense2, det-counterfactual, act 10.8) …sal relationships: Xevo has a direct effect on gyzp. Tijv has a direct effect on gyzp. We know that xevo or tijv causes gyzp. We observed an individual is tijv. | Would an individual is not gyzp if not xevo instead of xevo?

**Feature 6039** — AUROC discovery 0.67, held-out stories 0.66; fires on 53% of confounding vs 33% of others; top-8 purity 12%

- (diamondcut, nonsense8, det-counterfactual, act 13.1) …rect effect on rukz. Kraz has a direct effect on rukz. We know that hwax causes not pexu and kraz. pexu and kraz causes rukz. We observed an individual is hwax. | Would an individual is not rukz if not pexu instead of pexu?
- (diamondcut, nonsense8, det-counterfactual, act 13.1) …irect effect on rukz. Kraz has a direct effect on rukz. We know that hwax causes not pexu and kraz. pexu or kraz causes rukz. We observed an individual is hwax. | Would an individual is rukz if pexu instead of not pexu?
- (diamondcut, nonsense4, backadj, act 13.0) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to gyzp. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense4, backadj, act 13.0) …ct effect on lirg. Method 1: We look directly at how jyka correlates with lirg in general. Method 2: We look at this correlation case by case according to gyzp. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (diamondcut, nonsense3, det-counterfactual, act 12.5) …irect effect on xevu. Cwoi has a direct effect on xevu. We know that gwet causes not yomx and cwoi. yomx or cwoi causes xevu. We observed an individual is gwet. | Would an individual is xevu if not yomx instead of yomx?

**Feature 13946** — AUROC discovery 0.66, held-out stories 0.55; fires on 52% of confounding vs 32% of others; top-8 purity 0%

- (diamondcut, nonsense5, backadj, act 9.2) …ct effect on kwox. Method 1: We look directly at how jyka correlates with kwox in general. Method 2: We look at this correlation case by case according to qwiu. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (frontdoor, nonsense8, ate, act 9.0) …u and are not hwax, the probability of rukz is 37%. For those who are pexu and are hwax, the probability of rukz is 28%. The overall probability of pexu is 33%. | Will pexu decrease the chance of rukz?
- (frontdoor, nonsense8, nie, act 8.7) …u and are not hwax, the probability of rukz is 76%. For those who are pexu and are hwax, the probability of rukz is 40%. The overall probability of pexu is 39%. | Does pexu positively affect rukz through hwax?
- (frontdoor, nonsense8, ate, act 8.7) …u and are not hwax, the probability of rukz is 10%. For those who are pexu and are hwax, the probability of rukz is 85%. The overall probability of pexu is 59%. | Will pexu increase the chance of rukz?
- (frontdoor, nonsense8, nie, act 8.4) …u and are not hwax, the probability of rukz is 46%. For those who are pexu and are hwax, the probability of rukz is 46%. The overall probability of pexu is 44%. | Does pexu negatively affect rukz through hwax?

## Layer 12 · premise_end · structure = **collision**


**Feature 13295** — AUROC discovery 0.75, held-out stories 0.71; fires on 80% of collision vs 40% of others; top-8 purity 12%

- (collision, nonsense1, exp_away, act 23.9) …ity of kwoz is 96%. For those who are kwox and are not swoq, the probability of kwoz is 74%. For those who are kwox and are swoq, the probability of kwoz is 6%. | If we look at those who are swoq, does the chance of kwoz increase when kwox?
- (frontdoor, nonsense7, ett, act 22.1) …nd are wibl, the probability of uvzi is 78%. For those who are not zuph, the probability of wibl is 93%. For those who are zuph, the probability of wibl is 15%. | For those who are zuph, would it be less likely to see uvzi if the individual was not zuph?
- (frontdoor, nonsense6, ett, act 22.0) …nd are zupj, the probability of muvq is 35%. For those who are not yupt, the probability of zupj is 68%. For those who are yupt, the probability of zupj is 11%. | For those who are yupt, would it be more likely to see muvq if the individual was not yupt?
- (diamondcut, nonsense6, ett, act 21.2) …nd are xyfo, the probability of muvq is 70%. For those who are not yupt, the probability of xyfo is 15%. For those who are yupt, the probability of xyfo is 15%. | For those who are yupt, would it be more likely to see muvq if the individual was not yupt?
- (frontdoor, nonsense1, ett, act 20.3) …nd are swoq, the probability of kwoz is 67%. For those who are not kwox, the probability of swoq is 86%. For those who are kwox, the probability of swoq is 50%. | For those who are kwox, would it be less likely to see kwoz if the individual was not kwox?

**Feature 12800** — AUROC discovery 0.74, held-out stories 0.65; fires on 56% of collision vs 18% of others; top-8 purity 38%

- (frontdoor, nonsense7, ett, act 7.1) …nd are wibl, the probability of uvzi is 78%. For those who are not zuph, the probability of wibl is 93%. For those who are zuph, the probability of wibl is 15%. | For those who are zuph, would it be less likely to see uvzi if the individual was not zuph?
- (frontdoor, nonsense4, ett, act 6.9) …nd are hwax, the probability of lirg is 82%. For those who are not jyka, the probability of hwax is 86%. For those who are jyka, the probability of hwax is 23%. | For those who are jyka, would it be less likely to see lirg if the individual was not jyka?
- (mediation, nonsense4, nie, act 6.8) …nd are hwax, the probability of lirg is 82%. For those who are not jyka, the probability of hwax is 95%. For those who are jyka, the probability of hwax is 46%. | Does jyka positively affect lirg through hwax?
- (frontdoor, nonsense7, ett, act 6.7) …nd are wibl, the probability of uvzi is 89%. For those who are not zuph, the probability of wibl is 83%. For those who are zuph, the probability of wibl is 63%. | For those who are zuph, would it be more likely to see uvzi if the individual was not zuph?
- (collision, nonsense2, marginal, act 6.5) …ijv. The overall probability of xevo is 62%. For those who are not xevo, the probability of gyzp is 65%. For those who are xevo, the probability of gyzp is 65%. | Is gyzp more likely than not gyzp overall?

**Feature 5999** — AUROC discovery 0.74, held-out stories 0.61; fires on 41% of collision vs 7% of others; top-8 purity 50%

- (collision, nonsense4, collider_bias, act 4.9) …causal relationships: Jyka has a direct effect on hwax. Lirg has a direct effect on hwax. For those who are hwax, the correlation between jyka and lirg is 0.17. | If we look at those who are hwax, does it mean that jyka does not affect lirg?
- (collision, nonsense3, collider_bias, act 4.9) …ausal relationships: Yomx has a direct effect on gwet. Xevu has a direct effect on gwet. For those who are gwet, the correlation between yomx and xevu is -0.07. | If we look at those who are gwet, does it mean that yomx affects xevu?
- (frontdoor, nonsense8, marginal, act 4.7) …ved. The overall probability of pexu is 68%. For those who are not pexu, the probability of rukz is 78%. For those who are pexu, the probability of rukz is 79%. | Is rukz more likely than not rukz overall?
- (frontdoor, nonsense8, marginal, act 4.7) …ved. The overall probability of pexu is 59%. For those who are not pexu, the probability of rukz is 41%. For those who are pexu, the probability of rukz is 46%. | Is rukz less likely than not rukz overall?
- (collision, nonsense4, collider_bias, act 4.6) …causal relationships: Jyka has a direct effect on hwax. Lirg has a direct effect on hwax. For those who are hwax, the correlation between jyka and lirg is 0.04. | If we look at those who are hwax, does it mean that jyka affects lirg?

**Feature 9562** — AUROC discovery 0.73, held-out stories 0.50; fires on 46% of collision vs 35% of others; top-8 purity 100%

- (collision, nonsense1, marginal, act 10.5) … swoq. The overall probability of kwox is 32%. For those who are not kwox, the probability of kwoz is 7%. For those who are kwox, the probability of kwoz is 7%. | Is kwoz less likely than not kwoz overall?
- (collision, nonsense1, marginal, act 9.9) …woq. The overall probability of kwox is 97%. For those who are not kwox, the probability of kwoz is 49%. For those who are kwox, the probability of kwoz is 49%. | Is kwoz less likely than not kwoz overall?
- (collision, nonsense9, marginal, act 9.9) …yka. The overall probability of zuph is 59%. For those who are not zuph, the probability of glimx is 7%. For those who are zuph, the probability of glimx is 7%. | Is glimx more likely than not glimx overall?
- (collision, nonsense4, marginal, act 9.7) …wax. The overall probability of jyka is 59%. For those who are not jyka, the probability of lirg is 11%. For those who are jyka, the probability of lirg is 11%. | Is lirg less likely than not lirg overall?
- (collision, nonsense2, marginal, act 9.7) … tijv. The overall probability of xevo is 14%. For those who are not xevo, the probability of gyzp is 6%. For those who are xevo, the probability of gyzp is 6%. | Is gyzp more likely than not gyzp overall?

**Feature 1957** — AUROC discovery 0.72, held-out stories 0.61; fires on 37% of collision vs 5% of others; top-8 purity 88%

- (collision, nonsense3, marginal, act 5.5) … gwet. The overall probability of yomx is 36%. For those who are not yomx, the probability of xevu is 1%. For those who are yomx, the probability of xevu is 1%. | Is xevu less likely than not xevu overall?
- (collision, nonsense9, exp_away, act 5.4) …of glimx is 15%. For those who are zuph and are not jyka, the probability of glimx is 49%. For those who are zuph and are jyka, the probability of glimx is 14%. | If we look at those who are jyka, does the chance of glimx decrease when zuph?
- (collision, nonsense1, marginal, act 5.4) … swoq. The overall probability of kwox is 32%. For those who are not kwox, the probability of kwoz is 7%. For those who are kwox, the probability of kwoz is 7%. | Is kwoz less likely than not kwoz overall?
- (collision, nonsense2, exp_away, act 5.1) …ty of gyzp is 80%. For those who are xevo and are not tijv, the probability of gyzp is 52%. For those who are xevo and are tijv, the probability of gyzp is 32%. | If we look at those who are tijv, does the chance of gyzp increase when xevo?
- (mediation, nonsense6, det-counterfactual, act 5.1) … or causal relationships: Yupt has a direct effect on zupj and muvq. Zupj has a direct effect on muvq. We know that yupt causes zupj. yupt and zupj causes muvq. | Would an individual is muvq if not yupt instead of yupt?

## Layer 12 · premise_end · structure = **chain**


**Feature 2620** — AUROC discovery 0.82, held-out stories 0.86; fires on 100% of chain vs 89% of others; top-8 purity 25%

- (collision, nonsense8, collider_bias, act 21.8) …ausal relationships: Pexu has a direct effect on hwax. Rukz has a direct effect on hwax. For those who are hwax, the correlation between pexu and rukz is -0.00. | If we look at those who are hwax, does it mean that pexu affects rukz?
- (chain, nonsense2, det-counterfactual, act 21.6) … causal relationships: Xevo has a direct effect on tijv. Tijv has a direct effect on gyzp. We know that xevo causes not tijv, and we know that tijv causes gyzp. | Would an individual is gyzp if xevo instead of not xevo?
- (collision, nonsense3, collider_bias, act 21.5) …ausal relationships: Yomx has a direct effect on gwet. Xevu has a direct effect on gwet. For those who are gwet, the correlation between yomx and xevu is -0.07. | If we look at those who are gwet, does it mean that yomx affects xevu?
- (chain, nonsense3, det-counterfactual, act 21.5) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (mediation, nonsense5, det-counterfactual, act 21.5) … or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka and yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?

**Feature 11137** — AUROC discovery 0.77, held-out stories 0.76; fires on 82% of chain vs 43% of others; top-8 purity 50%

- (chain, nonsense0, ett, act 8.8) …t on zuph. Zuph has a direct effect on xevu. For those who are not rixq, the probability of xevu is 52%. For those who are rixq, the probability of xevu is 50%. | For those who are rixq, would it be less likely to see xevu if the individual was not rixq?
- (chain, nonsense2, nie, act 8.3) …t on tijv. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 50%. For those who are xevo, the probability of gyzp is 75%. | Does xevo negatively affect gyzp through tijv?
- (fork, nonsense3, ett, act 8.2) …t on xevu. Gwet has a direct effect on xevu. For those who are not yomx, the probability of xevu is 77%. For those who are yomx, the probability of xevu is 70%. | For those who are yomx, would it be less likely to see xevu if the individual was not yomx?
- (fork, nonsense8, ate, act 8.1) …t on rukz. Hwax has a direct effect on rukz. For those who are not pexu, the probability of rukz is 81%. For those who are pexu, the probability of rukz is 51%. | Will pexu decrease the chance of rukz?
- (chain, nonsense8, ate, act 8.0) …t on hwax. Hwax has a direct effect on rukz. For those who are not pexu, the probability of rukz is 71%. For those who are pexu, the probability of rukz is 72%. | Will pexu decrease the chance of rukz?

**Feature 14411** — AUROC discovery 0.74, held-out stories 0.70; fires on 99% of chain vs 84% of others; top-8 purity 38%

- (arrowhead, nonsense5, ate, act 16.5) …a direct effect on kwox. Qwiu is unobserved. For those who are not jyka, the probability of kwox is 51%. For those who are jyka, the probability of kwox is 72%. | Will jyka increase the chance of kwox?
- (mediation, nonsense9, ett, act 16.4) … glimx. Jyka has a direct effect on glimx. For those who are not zuph, the probability of glimx is 55%. For those who are zuph, the probability of glimx is 72%. | For those who are zuph, would it be more likely to see glimx if the individual was not zuph?
- (diamond, nonsense5, ate, act 15.8) …t on kwox. Qwiu has a direct effect on kwox. For those who are not jyka, the probability of kwox is 55%. For those who are jyka, the probability of kwox is 84%. | Will jyka increase the chance of kwox?
- (chain, nonsense2, nie, act 15.8) …t on tijv. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 50%. For those who are xevo, the probability of gyzp is 75%. | Does xevo negatively affect gyzp through tijv?
- (diamond, nonsense3, ett, act 15.8) …t on xevu. Cwoi has a direct effect on xevu. For those who are not yomx, the probability of xevu is 12%. For those who are yomx, the probability of xevu is 48%. | For those who are yomx, would it be more likely to see xevu if the individual was not yomx?

**Feature 304** — AUROC discovery 0.73, held-out stories 0.67; fires on 100% of chain vs 100% of others; top-8 purity 0%

- (IV, nonsense3, det-counterfactual, act 18.8) …t on xevu. Cwoi is unobserved. We know that cwoi or gwet causes yomx. cwoi or yomx causes xevu. We observed an individual is gwet and an individual is not cwoi. | Would an individual is xevu if not yomx instead of yomx?
- (IV, nonsense8, det-counterfactual, act 18.5) … on rukz. Kraz is unobserved. We know that kraz or hwax causes pexu. kraz and pexu causes rukz. We observed an individual is hwax and an individual is not kraz. | Would an individual is rukz if not pexu instead of pexu?
- (IV, nonsense1, det-counterfactual, act 18.5) …t on kwoz. Muvy is unobserved. We know that muvy or swoq causes kwox. muvy or kwox causes kwoz. We observed an individual is swoq and an individual is not muvy. | Would an individual is not kwoz if not kwox instead of kwox?
- (IV, nonsense2, det-counterfactual, act 18.2) …t on gyzp. Tijw is unobserved. We know that tijw or tijv causes xevo. tijw or xevo causes gyzp. We observed an individual is not tijv and an individual is tijw. | Would an individual is gyzp if xevo instead of not xevo?
- (IV, nonsense7, det-counterfactual, act 18.1) … uvzi. Vubr is unobserved. We know that vubr or wibl causes zuph. vubr or zuph causes uvzi. We observed an individual is not wibl and an individual is not vubr. | Would an individual is not uvzi if not zuph instead of zuph?

**Feature 13897** — AUROC discovery 0.72, held-out stories 0.64; fires on 74% of chain vs 45% of others; top-8 purity 12%

- (mediation, nonsense1, det-counterfactual, act 12.6) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is not kwoz if kwox instead of not kwox?
- (mediation, nonsense1, det-counterfactual, act 12.6) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is kwoz if kwox instead of not kwox?
- (mediation, nonsense7, det-counterfactual, act 11.5) … or causal relationships: Zuph has a direct effect on wibl and uvzi. Wibl has a direct effect on uvzi. We know that zuph causes wibl. zuph and wibl causes uvzi. | Would an individual is uvzi if zuph instead of not zuph?
- (mediation, nonsense8, det-counterfactual, act 11.3) … or causal relationships: Pexu has a direct effect on hwax and rukz. Hwax has a direct effect on rukz. We know that pexu causes hwax. pexu and hwax causes rukz. | Would an individual is not rukz if pexu instead of not pexu?
- (mediation, nonsense6, det-counterfactual, act 9.8) … or causal relationships: Yupt has a direct effect on zupj and muvq. Zupj has a direct effect on muvq. We know that yupt causes zupj. yupt and zupj causes muvq. | Would an individual is muvq if not yupt instead of yupt?

## Layer 12 · premise_end · structure = **fork**


**Feature 2620** — AUROC discovery 0.85, held-out stories 0.81; fires on 100% of fork vs 90% of others; top-8 purity 0%

- (collision, nonsense8, collider_bias, act 21.8) …ausal relationships: Pexu has a direct effect on hwax. Rukz has a direct effect on hwax. For those who are hwax, the correlation between pexu and rukz is -0.00. | If we look at those who are hwax, does it mean that pexu affects rukz?
- (chain, nonsense2, det-counterfactual, act 21.6) … causal relationships: Xevo has a direct effect on tijv. Tijv has a direct effect on gyzp. We know that xevo causes not tijv, and we know that tijv causes gyzp. | Would an individual is gyzp if xevo instead of not xevo?
- (collision, nonsense3, collider_bias, act 21.5) …ausal relationships: Yomx has a direct effect on gwet. Xevu has a direct effect on gwet. For those who are gwet, the correlation between yomx and xevu is -0.07. | If we look at those who are gwet, does it mean that yomx affects xevu?
- (chain, nonsense3, det-counterfactual, act 21.5) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (mediation, nonsense5, det-counterfactual, act 21.5) … or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka and yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?

**Feature 11137** — AUROC discovery 0.76, held-out stories 0.65; fires on 76% of fork vs 45% of others; top-8 purity 25%

- (chain, nonsense0, ett, act 8.8) …t on zuph. Zuph has a direct effect on xevu. For those who are not rixq, the probability of xevu is 52%. For those who are rixq, the probability of xevu is 50%. | For those who are rixq, would it be less likely to see xevu if the individual was not rixq?
- (chain, nonsense2, nie, act 8.3) …t on tijv. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 50%. For those who are xevo, the probability of gyzp is 75%. | Does xevo negatively affect gyzp through tijv?
- (fork, nonsense3, ett, act 8.2) …t on xevu. Gwet has a direct effect on xevu. For those who are not yomx, the probability of xevu is 77%. For those who are yomx, the probability of xevu is 70%. | For those who are yomx, would it be less likely to see xevu if the individual was not yomx?
- (fork, nonsense8, ate, act 8.1) …t on rukz. Hwax has a direct effect on rukz. For those who are not pexu, the probability of rukz is 81%. For those who are pexu, the probability of rukz is 51%. | Will pexu decrease the chance of rukz?
- (chain, nonsense8, ate, act 8.0) …t on hwax. Hwax has a direct effect on rukz. For those who are not pexu, the probability of rukz is 71%. For those who are pexu, the probability of rukz is 72%. | Will pexu decrease the chance of rukz?

**Feature 7417** — AUROC discovery 0.74, held-out stories 0.67; fires on 66% of fork vs 30% of others; top-8 purity 0%

- (chain, nonsense1, det-counterfactual, act 8.5) … causal relationships: Kwox has a direct effect on swoq. Swoq has a direct effect on kwoz. We know that kwox causes swoq, and we know that swoq causes not kwoz. | Would an individual is kwoz if kwox instead of not kwox?
- (chain, nonsense0, det-counterfactual, act 8.2) … causal relationships: Rixq has a direct effect on zuph. Zuph has a direct effect on xevu. We know that rixq causes zuph, and we know that zuph causes not xevu. | Would an individual is xevu if not rixq instead of rixq?
- (chain, nonsense5, det-counterfactual, act 8.2) … causal relationships: Jyka has a direct effect on yupt. Yupt has a direct effect on kwox. We know that jyka causes yupt, and we know that yupt causes not kwox. | Would an individual is kwox if not jyka instead of jyka?
- (chain, nonsense5, det-counterfactual, act 8.2) … causal relationships: Jyka has a direct effect on yupt. Yupt has a direct effect on kwox. We know that jyka causes yupt, and we know that yupt causes not kwox. | Would an individual is kwox if jyka instead of not jyka?
- (chain, nonsense6, det-counterfactual, act 8.0) … causal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes zupj, and we know that zupj causes not muvq. | Would an individual is not muvq if yupt instead of not yupt?

**Feature 7541** — AUROC discovery 0.73, held-out stories 0.73; fires on 100% of fork vs 98% of others; top-8 purity 50%

- (fork, nonsense5, marginal, act 33.4) …kwox. The overall probability of jyka is 54%. For those who are not jyka, the probability of kwox is 43%. For those who are jyka, the probability of kwox is 8%. | Is kwox more likely than not kwox overall?
- (confounding, nonsense4, marginal, act 32.4) …lirg. The overall probability of jyka is 17%. For those who are not jyka, the probability of lirg is 46%. For those who are jyka, the probability of lirg is 8%. | Is lirg more likely than not lirg overall?
- (fork, nonsense2, marginal, act 29.4) …yzp. The overall probability of xevo is 67%. For those who are not xevo, the probability of gyzp is 32%. For those who are xevo, the probability of gyzp is 11%. | Is gyzp more likely than not gyzp overall?
- (confounding, nonsense2, correlation, act 28.8) … Xevo has a direct effect on gyzp. The overall probability of xevo is 45%. The probability of not xevo and gyzp is 45%. The probability of xevo and gyzp is 11%. | Is the chance of gyzp larger when observing xevo?
- (mediation, nonsense2, ate, act 28.5) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?

**Feature 11518** — AUROC discovery 0.71, held-out stories 0.62; fires on 87% of fork vs 71% of others; top-8 purity 12%

- (IV, nonsense9, det-counterfactual, act 9.7) …ect on glimx. Zory is unobserved. We know that zory or jyka causes zuph. zory or zuph causes glimx. We observed an individual is jyka and an individual is zory. | Would an individual is glimx if zuph instead of not zuph?
- (IV, nonsense4, det-counterfactual, act 9.2) …fect on lirg. Gyzp is unobserved. We know that gyzp and hwax causes jyka. gyzp or jyka causes lirg. We observed an individual is hwax and an individual is gyzp. | Would an individual is lirg if jyka instead of not jyka?
- (IV, nonsense2, det-counterfactual, act 8.9) …t on gyzp. Tijw is unobserved. We know that tijw or tijv causes xevo. tijw or xevo causes gyzp. We observed an individual is not tijv and an individual is tijw. | Would an individual is gyzp if xevo instead of not xevo?
- (confounding, nonsense4, det-counterfactual, act 8.9) …irect effect on jyka and lirg. Jyka has a direct effect on lirg. We know that hwax causes jyka. hwax or jyka causes lirg. We observed an individual is not hwax. | Would an individual is lirg if not jyka instead of jyka?
- (IV, nonsense1, det-counterfactual, act 8.7) …t on kwoz. Muvy is unobserved. We know that muvy or swoq causes kwox. muvy or kwox causes kwoz. We observed an individual is swoq and an individual is not muvy. | Would an individual is not kwoz if not kwox instead of kwox?

## Layer 12 · premise_end · structure = **mediation**


**Feature 9257** — AUROC discovery 0.72, held-out stories 0.61; fires on 68% of mediation vs 48% of others; top-8 purity 0%

- (arrowhead, nonsense4, nie, act 15.6) …a and are not gyzp, the probability of hwax is 93%. For those who are jyka and are gyzp, the probability of hwax is 48%. The overall probability of gyzp is 34%. | Does jyka positively affect lirg through hwax?
- (arrowhead, nonsense4, nie, act 15.3) …a and are not gyzp, the probability of hwax is 13%. For those who are jyka and are gyzp, the probability of hwax is 96%. The overall probability of gyzp is 18%. | Does jyka negatively affect lirg through hwax?
- (arrowhead, nonsense4, nie, act 15.2) …a and are not gyzp, the probability of hwax is 74%. For those who are jyka and are gyzp, the probability of hwax is 75%. The overall probability of gyzp is 70%. | Does jyka positively affect lirg through hwax?
- (arrowhead, nonsense8, nde, act 12.9) …xu and are not kraz, the probability of hwax is 89%. For those who are pexu and are kraz, the probability of hwax is 71%. The overall probability of kraz is 4%. | If we disregard the mediation effect through hwax, would pexu positively affect rukz?
- (arrowhead, nonsense4, nie, act 12.3) …ka and are not gyzp, the probability of hwax is 81%. For those who are jyka and are gyzp, the probability of hwax is 42%. The overall probability of gyzp is 7%. | Does jyka positively affect lirg through hwax?

**Feature 7815** — AUROC discovery 0.71, held-out stories 0.67; fires on 72% of mediation vs 38% of others; top-8 purity 0%

- (chain, nonsense5, det-counterfactual, act 14.4) … causal relationships: Jyka has a direct effect on yupt. Yupt has a direct effect on kwox. We know that jyka causes not yupt, and we know that yupt causes kwox. | Would an individual is kwox if jyka instead of not jyka?
- (chain, nonsense0, det-counterfactual, act 14.3) … causal relationships: Rixq has a direct effect on zuph. Zuph has a direct effect on xevu. We know that rixq causes not zuph, and we know that zuph causes xevu. | Would an individual is not xevu if not rixq instead of rixq?
- (chain, nonsense0, det-counterfactual, act 13.8) …s or causal relationships: Rixq has a direct effect on zuph. Zuph has a direct effect on xevu. We know that rixq causes zuph, and we know that zuph causes xevu. | Would an individual is xevu if not rixq instead of rixq?
- (chain, nonsense7, det-counterfactual, act 13.8) …s or causal relationships: Zuph has a direct effect on wibl. Wibl has a direct effect on uvzi. We know that zuph causes wibl, and we know that wibl causes uvzi. | Would an individual is uvzi if zuph instead of not zuph?
- (chain, nonsense7, det-counterfactual, act 13.6) … causal relationships: Zuph has a direct effect on wibl. Wibl has a direct effect on uvzi. We know that zuph causes not wibl, and we know that wibl causes uvzi. | Would an individual is not uvzi if not zuph instead of zuph?

**Feature 12003** — AUROC discovery 0.69, held-out stories 0.54; fires on 69% of mediation vs 49% of others; top-8 purity 12%

- (frontdoor, nonsense5, correlation, act 18.0) …effect on kwox. Qwiu is unobserved. The overall probability of jyka is 88%. The probability of not jyka and kwox is 8%. The probability of jyka and kwox is 65%. | Is the chance of kwox smaller when observing jyka?
- (diamondcut, nonsense8, correlation, act 17.3) … Kraz has a direct effect on rukz. The overall probability of pexu is 64%. The probability of not pexu and rukz is 20%. The probability of pexu and rukz is 62%. | Is the chance of rukz smaller when observing pexu?
- (frontdoor, nonsense5, correlation, act 17.1) …ffect on kwox. Qwiu is unobserved. The overall probability of jyka is 82%. The probability of not jyka and kwox is 14%. The probability of jyka and kwox is 50%. | Is the chance of kwox larger when observing jyka?
- (frontdoor, nonsense5, correlation, act 17.1) …ffect on kwox. Qwiu is unobserved. The overall probability of jyka is 82%. The probability of not jyka and kwox is 14%. The probability of jyka and kwox is 50%. | Is the chance of kwox smaller when observing jyka?
- (mediation, nonsense2, correlation, act 17.1) …. Tijv has a direct effect on gyzp. The overall probability of xevo is 84%. The probability of not xevo and gyzp is 1%. The probability of xevo and gyzp is 73%. | Is the chance of gyzp smaller when observing xevo?

**Feature 4571** — AUROC discovery 0.69, held-out stories 0.58; fires on 90% of mediation vs 66% of others; top-8 purity 0%

- (confounding, nonsense1, correlation, act 17.7) … Kwox has a direct effect on kwoz. The overall probability of kwox is 65%. The probability of not kwox and kwoz is 16%. The probability of kwox and kwoz is 50%. | Is the chance of kwoz larger when observing kwox?
- (fork, nonsense5, correlation, act 17.7) …. Yupt has a direct effect on kwox. The overall probability of jyka is 71%. The probability of not jyka and kwox is 5%. The probability of jyka and kwox is 65%. | Is the chance of kwox smaller when observing jyka?
- (confounding, nonsense8, correlation, act 17.5) … Pexu has a direct effect on rukz. The overall probability of pexu is 67%. The probability of not pexu and rukz is 15%. The probability of pexu and rukz is 54%. | Is the chance of rukz smaller when observing pexu?
- (IV, nonsense8, correlation, act 17.5) …ffect on rukz. Kraz is unobserved. The overall probability of pexu is 61%. The probability of not pexu and rukz is 26%. The probability of pexu and rukz is 29%. | Is the chance of rukz larger when observing pexu?
- (diamond, nonsense3, correlation, act 17.3) … Cwoi has a direct effect on xevu. The overall probability of yomx is 72%. The probability of not yomx and xevu is 15%. The probability of yomx and xevu is 35%. | Is the chance of xevu smaller when observing yomx?

**Feature 4822** — AUROC discovery 0.69, held-out stories 0.62; fires on 39% of mediation vs 10% of others; top-8 purity 0%

- (chain, nonsense2, det-counterfactual, act 9.2) … causal relationships: Xevo has a direct effect on tijv. Tijv has a direct effect on gyzp. We know that xevo causes tijv, and we know that tijv causes not gyzp. | Would an individual is gyzp if xevo instead of not xevo?
- (chain, nonsense8, det-counterfactual, act 8.9) … causal relationships: Pexu has a direct effect on hwax. Hwax has a direct effect on rukz. We know that pexu causes hwax, and we know that hwax causes not rukz. | Would an individual is not rukz if pexu instead of not pexu?
- (chain, nonsense3, det-counterfactual, act 8.8) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (chain, nonsense5, det-counterfactual, act 8.2) … causal relationships: Jyka has a direct effect on yupt. Yupt has a direct effect on kwox. We know that jyka causes yupt, and we know that yupt causes not kwox. | Would an individual is kwox if jyka instead of not jyka?
- (chain, nonsense5, det-counterfactual, act 8.2) … causal relationships: Jyka has a direct effect on yupt. Yupt has a direct effect on kwox. We know that jyka causes yupt, and we know that yupt causes not kwox. | Would an individual is kwox if not jyka instead of jyka?

## Layer 18 · premise_end · structure = **confounding**


**Feature 3407** — AUROC discovery 0.64, held-out stories 0.67; fires on 51% of confounding vs 24% of others; top-8 purity 25%

- (fork, nonsense1, det-counterfactual, act 17.2) …al relationships: Kwox has a direct effect on kwoz. Swoq has a direct effect on kwoz. We know that kwox and swoq causes kwoz. We observed an individual is swoq. | Would an individual is kwoz if not kwox instead of kwox?
- (fork, nonsense3, det-counterfactual, act 17.2) …al relationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is gwet. | Would an individual is not xevu if yomx instead of not yomx?
- (fork, nonsense3, det-counterfactual, act 17.2) …al relationships: Yomx has a direct effect on xevu. Gwet has a direct effect on xevu. We know that yomx and gwet causes xevu. We observed an individual is gwet. | Would an individual is not xevu if not yomx instead of yomx?
- (fork, nonsense2, det-counterfactual, act 16.8) …sal relationships: Xevo has a direct effect on gyzp. Tijv has a direct effect on gyzp. We know that xevo or tijv causes gyzp. We observed an individual is tijv. | Would an individual is not gyzp if not xevo instead of xevo?
- (confounding, nonsense1, det-counterfactual, act 16.1) …a direct effect on kwox and kwoz. Kwox has a direct effect on kwoz. We know that swoq causes kwox. swoq and kwox causes kwoz. We observed an individual is swoq. | Would an individual is kwoz if kwox instead of not kwox?

**Feature 15919** — AUROC discovery 0.64, held-out stories 0.63; fires on 39% of confounding vs 13% of others; top-8 purity 0%

- (frontdoor, nonsense1, nie, act 11.6) …x and are not swoq, the probability of kwoz is 35%. For those who are kwox and are swoq, the probability of kwoz is 41%. The overall probability of kwox is 47%. | Does kwox positively affect kwoz through swoq?
- (frontdoor, nonsense1, nie, act 11.5) …x and are not swoq, the probability of kwoz is 10%. For those who are kwox and are swoq, the probability of kwoz is 79%. The overall probability of kwox is 70%. | Does kwox negatively affect kwoz through swoq?
- (frontdoor, nonsense1, nie, act 11.5) …x and are not swoq, the probability of kwoz is 10%. For those who are kwox and are swoq, the probability of kwoz is 79%. The overall probability of kwox is 70%. | Does kwox positively affect kwoz through swoq?
- (frontdoor, nonsense1, nie, act 11.3) …x and are not swoq, the probability of kwoz is 92%. For those who are kwox and are swoq, the probability of kwoz is 22%. The overall probability of kwox is 51%. | Does kwox positively affect kwoz through swoq?
- (frontdoor, nonsense1, nie, act 11.2) …x and are not swoq, the probability of kwoz is 67%. For those who are kwox and are swoq, the probability of kwoz is 53%. The overall probability of kwox is 53%. | Does kwox negatively affect kwoz through swoq?

**Feature 3725** — AUROC discovery 0.63, held-out stories 0.61; fires on 59% of confounding vs 39% of others; top-8 purity 25%

- (mediation, nonsense2, ate, act 23.0) …v and gyzp. Tijv has a direct effect on gyzp. For those who are not xevo, the probability of gyzp is 72%. For those who are xevo, the probability of gyzp is 7%. | Will xevo increase the chance of gyzp?
- (diamondcut, nonsense9, marginal, act 19.6) …x. The overall probability of zuph is 82%. For those who are not zuph, the probability of glimx is 65%. For those who are zuph, the probability of glimx is 11%. | Is glimx more likely than not glimx overall?
- (fork, nonsense5, marginal, act 19.5) …kwox. The overall probability of jyka is 54%. For those who are not jyka, the probability of kwox is 43%. For those who are jyka, the probability of kwox is 8%. | Is kwox more likely than not kwox overall?
- (IV, nonsense7, marginal, act 18.6) …rved. The overall probability of zuph is 63%. For those who are not zuph, the probability of uvzi is 30%. For those who are zuph, the probability of uvzi is 7%. | Is uvzi more likely than not uvzi overall?
- (confounding, nonsense8, ett, act 18.5) …ax and are not pexu, the probability of rukz is 58%. For those who are hwax and are pexu, the probability of rukz is 56%. The overall probability of hwax is 1%. | For those who are pexu, would it be more likely to see rukz if the individual was not pexu?

**Feature 3629** — AUROC discovery 0.63, held-out stories 0.58; fires on 49% of confounding vs 34% of others; top-8 purity 0%

- (arrowhead, nonsense5, nie, act 22.2) …ka and are not qwiu, the probability of yupt is 82%. For those who are jyka and are qwiu, the probability of yupt is 45%. The overall probability of qwiu is 1%. | Does jyka positively affect kwox through yupt?
- (frontdoor, nonsense8, nie, act 22.1) …u and are not hwax, the probability of rukz is 46%. For those who are pexu and are hwax, the probability of rukz is 46%. The overall probability of pexu is 44%. | Does pexu negatively affect rukz through hwax?
- (frontdoor, nonsense8, nie, act 20.9) …u and are not hwax, the probability of rukz is 76%. For those who are pexu and are hwax, the probability of rukz is 40%. The overall probability of pexu is 39%. | Does pexu positively affect rukz through hwax?
- (frontdoor, nonsense8, ate, act 20.7) …u and are not hwax, the probability of rukz is 10%. For those who are pexu and are hwax, the probability of rukz is 85%. The overall probability of pexu is 59%. | Will pexu increase the chance of rukz?
- (frontdoor, nonsense8, ate, act 20.4) …u and are not hwax, the probability of rukz is 37%. For those who are pexu and are hwax, the probability of rukz is 28%. The overall probability of pexu is 33%. | Will pexu decrease the chance of rukz?

**Feature 2152** — AUROC discovery 0.62, held-out stories 0.62; fires on 44% of confounding vs 23% of others; top-8 purity 25%

- (confounding, nonsense7, ett, act 11.5) …l and are not zuph, the probability of uvzi is 75%. For those who are wibl and are zuph, the probability of uvzi is 97%. The overall probability of wibl is 85%. | For those who are zuph, would it be less likely to see uvzi if the individual was not zuph?
- (collision, nonsense8, exp_away, act 11.5) …y of rukz is 99%. For those who are pexu and are not hwax, the probability of rukz is 92%. For those who are pexu and are hwax, the probability of rukz is 100%. | If we look at those who are hwax, does the chance of rukz decrease when pexu?
- (diamondcut, nonsense1, det-counterfactual, act 11.3) … a direct effect on kwoz. Muvy has a direct effect on kwoz. We know that swoq causes kwox and muvy. kwox or muvy causes kwoz. We observed an individual is swoq. | Would an individual is kwoz if not kwox instead of kwox?
- (IV, nonsense1, det-counterfactual, act 10.5) …t on kwoz. Muvy is unobserved. We know that muvy or swoq causes kwox. muvy or kwox causes kwoz. We observed an individual is swoq and an individual is not muvy. | Would an individual is not kwoz if not kwox instead of kwox?
- (collision, nonsense9, backadj, act 10.4) …ka. Method 1: We look at how zuph correlates with glimx case by case according to jyka. Method 2: We look directly at how zuph correlates with glimx in general. | To understand how zuph affects glimx, is it more correct to use the Method 1 than Method 2?

## Layer 18 · premise_end · structure = **collision**


**Feature 4602** — AUROC discovery 0.82, held-out stories 0.78; fires on 85% of collision vs 38% of others; top-8 purity 100%

- (collision, nonsense9, backadj, act 27.3) …ka. Method 1: We look at how zuph correlates with glimx case by case according to jyka. Method 2: We look directly at how zuph correlates with glimx in general. | To understand how zuph affects glimx, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense9, backadj, act 27.3) …ka. Method 1: We look at how zuph correlates with glimx case by case according to jyka. Method 2: We look directly at how zuph correlates with glimx in general. | To understand how zuph affects glimx, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense0, collider_bias, act 27.0) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.01. | If we look at those who are zuph, does it mean that rixq affects xevu?
- (collision, nonsense0, collider_bias, act 25.9) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.07. | If we look at those who are zuph, does it mean that rixq does not affect xevu?
- (collision, nonsense9, collider_bias, act 25.2) …usal relationships: Zuph has a direct effect on jyka. Glimx has a direct effect on jyka. For those who are jyka, the correlation between zuph and glimx is 0.07. | If we look at those who are jyka, does it mean that zuph affects glimx?

**Feature 13627** — AUROC discovery 0.81, held-out stories 0.66; fires on 63% of collision vs 30% of others; top-8 purity 100%

- (collision, nonsense9, collider_bias, act 17.9) …usal relationships: Zuph has a direct effect on jyka. Glimx has a direct effect on jyka. For those who are jyka, the correlation between zuph and glimx is 0.07. | If we look at those who are jyka, does it mean that zuph affects glimx?
- (collision, nonsense4, collider_bias, act 17.3) …causal relationships: Jyka has a direct effect on hwax. Lirg has a direct effect on hwax. For those who are hwax, the correlation between jyka and lirg is 0.17. | If we look at those who are hwax, does it mean that jyka does not affect lirg?
- (collision, nonsense3, collider_bias, act 17.1) …ausal relationships: Yomx has a direct effect on gwet. Xevu has a direct effect on gwet. For those who are gwet, the correlation between yomx and xevu is -0.07. | If we look at those who are gwet, does it mean that yomx affects xevu?
- (collision, nonsense6, collider_bias, act 16.8) …ausal relationships: Yupt has a direct effect on zupj. Muvq has a direct effect on zupj. For those who are zupj, the correlation between yupt and muvq is -0.07. | If we look at those who are zupj, does it mean that yupt affects muvq?
- (collision, nonsense4, collider_bias, act 16.1) …causal relationships: Jyka has a direct effect on hwax. Lirg has a direct effect on hwax. For those who are hwax, the correlation between jyka and lirg is 0.04. | If we look at those who are hwax, does it mean that jyka affects lirg?

**Feature 3462** — AUROC discovery 0.79, held-out stories 0.65; fires on 61% of collision vs 20% of others; top-8 purity 25%

- (diamondcut, nonsense8, backadj, act 16.1) …rukz. Method 1: We look at how pexu correlates with rukz case by case according to kraz. Method 2: We look directly at how pexu correlates with rukz in general. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense4, backadj, act 15.9) …hwax. Method 1: We look at how jyka correlates with lirg case by case according to hwax. Method 2: We look directly at how jyka correlates with lirg in general. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (collision, nonsense4, backadj, act 15.9) …hwax. Method 1: We look at how jyka correlates with lirg case by case according to hwax. Method 2: We look directly at how jyka correlates with lirg in general. | To understand how jyka affects lirg, is it more correct to use the Method 1 than Method 2?
- (mediation, nonsense8, backadj, act 15.6) …rukz. Method 1: We look at how pexu correlates with rukz case by case according to hwax. Method 2: We look directly at how pexu correlates with rukz in general. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (diamond, nonsense0, backadj, act 15.4) …xevu. Method 1: We look at how rixq correlates with xevu case by case according to swoy. Method 2: We look directly at how rixq correlates with xevu in general. | To understand how rixq affects xevu, is it more correct to use the Method 1 than Method 2?

**Feature 11833** — AUROC discovery 0.77, held-out stories 0.67; fires on 52% of collision vs 13% of others; top-8 purity 25%

- (IV, nonsense4, ate, act 17.1) …ho are hwax, the probability of lirg is 82%. For those who are not hwax, the probability of jyka is 48%. For those who are hwax, the probability of jyka is 33%. | Will jyka increase the chance of lirg?
- (collision, nonsense2, marginal, act 15.6) …tijv. The overall probability of xevo is 0%. For those who are not xevo, the probability of gyzp is 34%. For those who are xevo, the probability of gyzp is 34%. | Is gyzp more likely than not gyzp overall?
- (IV, nonsense6, ate, act 15.6) …ho are zupj, the probability of muvq is 29%. For those who are not zupj, the probability of yupt is 48%. For those who are zupj, the probability of yupt is 44%. | Will yupt increase the chance of muvq?
- (chain, nonsense2, det-counterfactual, act 15.1) …sal relationships: Xevo has a direct effect on tijv. Tijv has a direct effect on gyzp. We know that xevo causes not tijv, and we know that tijv causes not gyzp. | Would an individual is gyzp if not xevo instead of xevo?
- (IV, nonsense7, ate, act 15.1) …who are wibl, the probability of uvzi is 29%. For those who are not wibl, the probability of zuph is 5%. For those who are wibl, the probability of zuph is 28%. | Will zuph increase the chance of uvzi?

**Feature 782** — AUROC discovery 0.76, held-out stories 0.64; fires on 56% of collision vs 18% of others; top-8 purity 0%

- (chain, nonsense7, nie, act 37.2) …t on wibl. Wibl has a direct effect on uvzi. For those who are not zuph, the probability of uvzi is 31%. For those who are zuph, the probability of uvzi is 31%. | Does zuph positively affect uvzi through wibl?
- (chain, nonsense7, nie, act 36.1) …t on wibl. Wibl has a direct effect on uvzi. For those who are not zuph, the probability of uvzi is 97%. For those who are zuph, the probability of uvzi is 97%. | Does zuph positively affect uvzi through wibl?
- (diamond, nonsense3, ate, act 35.8) …t on xevu. Cwoi has a direct effect on xevu. For those who are not yomx, the probability of xevu is 45%. For those who are yomx, the probability of xevu is 45%. | Will yomx increase the chance of xevu?
- (chain, nonsense6, ate, act 35.2) …t on zupj. Zupj has a direct effect on muvq. For those who are not yupt, the probability of muvq is 25%. For those who are yupt, the probability of muvq is 25%. | Will yupt increase the chance of muvq?
- (chain, nonsense6, ett, act 35.2) …t on zupj. Zupj has a direct effect on muvq. For those who are not yupt, the probability of muvq is 25%. For those who are yupt, the probability of muvq is 25%. | For those who are yupt, would it be more likely to see muvq if the individual was not yupt?

## Layer 18 · premise_end · structure = **chain**


**Feature 9392** — AUROC discovery 0.80, held-out stories 0.82; fires on 100% of chain vs 66% of others; top-8 purity 0%

- (mediation, nonsense5, det-counterfactual, act 29.5) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is kwox if jyka instead of not jyka?
- (mediation, nonsense5, det-counterfactual, act 29.4) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is not kwox if not jyka instead of jyka?
- (mediation, nonsense5, det-counterfactual, act 29.4) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?
- (mediation, nonsense1, det-counterfactual, act 27.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is kwoz if kwox instead of not kwox?
- (mediation, nonsense1, det-counterfactual, act 27.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is not kwoz if kwox instead of not kwox?

**Feature 7503** — AUROC discovery 0.78, held-out stories 0.72; fires on 89% of chain vs 50% of others; top-8 purity 100%

- (chain, nonsense8, det-counterfactual, act 20.0) …s or causal relationships: Pexu has a direct effect on hwax. Hwax has a direct effect on rukz. We know that pexu causes hwax, and we know that hwax causes rukz. | Would an individual is rukz if not pexu instead of pexu?
- (chain, nonsense6, det-counterfactual, act 19.6) …s or causal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes zupj, and we know that zupj causes muvq. | Would an individual is not muvq if yupt instead of not yupt?
- (chain, nonsense3, det-counterfactual, act 19.5) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (chain, nonsense4, det-counterfactual, act 19.5) …s or causal relationships: Jyka has a direct effect on hwax. Hwax has a direct effect on lirg. We know that jyka causes hwax, and we know that hwax causes lirg. | Would an individual is not lirg if not jyka instead of jyka?
- (chain, nonsense4, det-counterfactual, act 19.5) …s or causal relationships: Jyka has a direct effect on hwax. Hwax has a direct effect on lirg. We know that jyka causes hwax, and we know that hwax causes lirg. | Would an individual is lirg if not jyka instead of jyka?

**Feature 11160** — AUROC discovery 0.76, held-out stories 0.73; fires on 90% of chain vs 54% of others; top-8 purity 25%

- (mediation, nonsense5, ate, act 18.0) … and kwox. Yupt has a direct effect on kwox. For those who are not jyka, the probability of kwox is 81%. For those who are jyka, the probability of kwox is 27%. | Will jyka decrease the chance of kwox?
- (chain, nonsense3, ett, act 17.8) …t on gwet. Gwet has a direct effect on xevu. For those who are not yomx, the probability of xevu is 68%. For those who are yomx, the probability of xevu is 40%. | For those who are yomx, would it be less likely to see xevu if the individual was not yomx?
- (diamond, nonsense5, nie, act 17.6) …t on kwox. Qwiu has a direct effect on kwox. For those who are not jyka, the probability of kwox is 31%. For those who are jyka, the probability of kwox is 33%. | Does jyka negatively affect kwox through yupt and qwiu?
- (diamond, nonsense5, nie, act 17.3) …t on kwox. Qwiu has a direct effect on kwox. For those who are not jyka, the probability of kwox is 51%. For those who are jyka, the probability of kwox is 47%. | Does jyka negatively affect kwox through yupt and qwiu?
- (chain, nonsense3, nie, act 17.1) …t on gwet. Gwet has a direct effect on xevu. For those who are not yomx, the probability of xevu is 74%. For those who are yomx, the probability of xevu is 36%. | Does yomx positively affect xevu through gwet?

**Feature 2779** — AUROC discovery 0.76, held-out stories 0.69; fires on 87% of chain vs 51% of others; top-8 purity 62%

- (mediation, nonsense8, backadj, act 34.3) …rukz. Method 1: We look at how pexu correlates with rukz case by case according to hwax. Method 2: We look directly at how pexu correlates with rukz in general. | To understand how pexu affects rukz, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense3, backadj, act 34.3) …xevu. Method 1: We look at how yomx correlates with xevu case by case according to gwet. Method 2: We look directly at how yomx correlates with xevu in general. | To understand how yomx affects xevu, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense5, backadj, act 33.5) …kwox. Method 1: We look at how jyka correlates with kwox case by case according to yupt. Method 2: We look directly at how jyka correlates with kwox in general. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense5, backadj, act 33.5) …kwox. Method 1: We look at how jyka correlates with kwox case by case according to yupt. Method 2: We look directly at how jyka correlates with kwox in general. | To understand how jyka affects kwox, is it more correct to use the Method 1 than Method 2?
- (chain, nonsense0, backadj, act 33.4) …xevu. Method 1: We look at how rixq correlates with xevu case by case according to zuph. Method 2: We look directly at how rixq correlates with xevu in general. | To understand how rixq affects xevu, is it more correct to use the Method 1 than Method 2?

**Feature 7856** — AUROC discovery 0.76, held-out stories 0.66; fires on 100% of chain vs 100% of others; top-8 purity 100%

- (chain, nonsense6, det-counterfactual, act 52.6) …s or causal relationships: Yupt has a direct effect on zupj. Zupj has a direct effect on muvq. We know that yupt causes zupj, and we know that zupj causes muvq. | Would an individual is not muvq if yupt instead of not yupt?
- (chain, nonsense7, det-counterfactual, act 52.3) …s or causal relationships: Zuph has a direct effect on wibl. Wibl has a direct effect on uvzi. We know that zuph causes wibl, and we know that wibl causes uvzi. | Would an individual is uvzi if zuph instead of not zuph?
- (chain, nonsense0, det-counterfactual, act 50.7) …s or causal relationships: Rixq has a direct effect on zuph. Zuph has a direct effect on xevu. We know that rixq causes zuph, and we know that zuph causes xevu. | Would an individual is xevu if not rixq instead of rixq?
- (chain, nonsense3, det-counterfactual, act 50.5) …s or causal relationships: Yomx has a direct effect on gwet. Gwet has a direct effect on xevu. We know that yomx causes gwet, and we know that gwet causes xevu. | Would an individual is not xevu if yomx instead of not yomx?
- (chain, nonsense7, backadj, act 48.7) …ct effect on uvzi. Method 1: We look directly at how zuph correlates with uvzi in general. Method 2: We look at this correlation case by case according to wibl. | To understand how zuph affects uvzi, is it more correct to use the Method 1 than Method 2?

## Layer 18 · premise_end · structure = **fork**


**Feature 9392** — AUROC discovery 0.84, held-out stories 0.84; fires on 100% of fork vs 67% of others; top-8 purity 0%

- (mediation, nonsense5, det-counterfactual, act 29.5) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is kwox if jyka instead of not jyka?
- (mediation, nonsense5, det-counterfactual, act 29.4) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is not kwox if not jyka instead of jyka?
- (mediation, nonsense5, det-counterfactual, act 29.4) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?
- (mediation, nonsense1, det-counterfactual, act 27.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is kwoz if kwox instead of not kwox?
- (mediation, nonsense1, det-counterfactual, act 27.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is not kwoz if kwox instead of not kwox?

**Feature 1645** — AUROC discovery 0.77, held-out stories 0.78; fires on 99% of fork vs 87% of others; top-8 purity 50%

- (fork, nonsense5, marginal, act 54.9) …kwox. The overall probability of jyka is 54%. For those who are not jyka, the probability of kwox is 43%. For those who are jyka, the probability of kwox is 8%. | Is kwox more likely than not kwox overall?
- (confounding, nonsense4, marginal, act 47.3) …lirg. The overall probability of jyka is 17%. For those who are not jyka, the probability of lirg is 46%. For those who are jyka, the probability of lirg is 8%. | Is lirg more likely than not lirg overall?
- (fork, nonsense2, marginal, act 46.8) …yzp. The overall probability of xevo is 67%. For those who are not xevo, the probability of gyzp is 32%. For those who are xevo, the probability of gyzp is 11%. | Is gyzp more likely than not gyzp overall?
- (IV, nonsense7, marginal, act 43.2) …rved. The overall probability of zuph is 63%. For those who are not zuph, the probability of uvzi is 30%. For those who are zuph, the probability of uvzi is 7%. | Is uvzi more likely than not uvzi overall?
- (confounding, nonsense2, correlation, act 41.2) …. Xevo has a direct effect on gyzp. The overall probability of xevo is 42%. The probability of not xevo and gyzp is 24%. The probability of xevo and gyzp is 9%. | Is the chance of gyzp smaller when observing xevo?

**Feature 4397** — AUROC discovery 0.73, held-out stories 0.68; fires on 88% of fork vs 60% of others; top-8 purity 0%

- (diamondcut, nonsense0, marginal, act 19.6) …evu. The overall probability of rixq is 28%. For those who are not rixq, the probability of xevu is 43%. For those who are rixq, the probability of xevu is 80%. | Is xevu more likely than not xevu overall?
- (diamondcut, nonsense3, marginal, act 19.1) …evu. The overall probability of yomx is 76%. For those who are not yomx, the probability of xevu is 83%. For those who are yomx, the probability of xevu is 93%. | Is xevu more likely than not xevu overall?
- (diamondcut, nonsense5, marginal, act 18.8) …wox. The overall probability of jyka is 56%. For those who are not jyka, the probability of kwox is 76%. For those who are jyka, the probability of kwox is 57%. | Is kwox less likely than not kwox overall?
- (diamondcut, nonsense4, marginal, act 18.7) …irg. The overall probability of jyka is 13%. For those who are not jyka, the probability of lirg is 55%. For those who are jyka, the probability of lirg is 66%. | Is lirg less likely than not lirg overall?
- (diamondcut, nonsense0, marginal, act 18.6) …evu. The overall probability of rixq is 78%. For those who are not rixq, the probability of xevu is 30%. For those who are rixq, the probability of xevu is 39%. | Is xevu less likely than not xevu overall?

**Feature 4500** — AUROC discovery 0.71, held-out stories 0.70; fires on 88% of fork vs 54% of others; top-8 purity 0%

- (chain, nonsense1, correlation, act 17.1) … Swoq has a direct effect on kwoz. The overall probability of kwox is 67%. The probability of not kwox and kwoz is 18%. The probability of kwox and kwoz is 57%. | Is the chance of kwoz larger when observing kwox?
- (chain, nonsense0, ett, act 17.1) …t on zuph. Zuph has a direct effect on xevu. For those who are not rixq, the probability of xevu is 32%. For those who are rixq, the probability of xevu is 70%. | For those who are rixq, would it be less likely to see xevu if the individual was not rixq?
- (confounding, nonsense8, correlation, act 16.6) … Pexu has a direct effect on rukz. The overall probability of pexu is 67%. The probability of not pexu and rukz is 15%. The probability of pexu and rukz is 54%. | Is the chance of rukz smaller when observing pexu?
- (chain, nonsense0, nie, act 16.4) …t on zuph. Zuph has a direct effect on xevu. For those who are not rixq, the probability of xevu is 35%. For those who are rixq, the probability of xevu is 66%. | Does rixq positively affect xevu through zuph?
- (diamond, nonsense1, ett, act 16.4) …t on kwoz. Muvy has a direct effect on kwoz. For those who are not kwox, the probability of kwoz is 39%. For those who are kwox, the probability of kwoz is 83%. | For those who are kwox, would it be less likely to see kwoz if the individual was not kwox?

**Feature 8693** — AUROC discovery 0.71, held-out stories 0.61; fires on 93% of fork vs 94% of others; top-8 purity 0%

- (collision, nonsense5, collider_bias, act 67.4) …ausal relationships: Jyka has a direct effect on yupt. Kwox has a direct effect on yupt. For those who are yupt, the correlation between jyka and kwox is -0.04. | If we look at those who are yupt, does it mean that jyka affects kwox?
- (collision, nonsense0, collider_bias, act 67.1) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.07. | If we look at those who are zuph, does it mean that rixq does not affect xevu?
- (collision, nonsense6, collider_bias, act 66.8) …ausal relationships: Yupt has a direct effect on zupj. Muvq has a direct effect on zupj. For those who are zupj, the correlation between yupt and muvq is -0.07. | If we look at those who are zupj, does it mean that yupt affects muvq?
- (collision, nonsense0, collider_bias, act 66.4) …ausal relationships: Rixq has a direct effect on zuph. Xevu has a direct effect on zuph. For those who are zuph, the correlation between rixq and xevu is -0.01. | If we look at those who are zuph, does it mean that rixq affects xevu?
- (collision, nonsense3, collider_bias, act 64.2) …ausal relationships: Yomx has a direct effect on gwet. Xevu has a direct effect on gwet. For those who are gwet, the correlation between yomx and xevu is -0.07. | If we look at those who are gwet, does it mean that yomx affects xevu?

## Layer 18 · premise_end · structure = **mediation**


**Feature 1555** — AUROC discovery 0.74, held-out stories 0.58; fires on 57% of mediation vs 29% of others; top-8 purity 0%

- (IV, nonsense0, ate, act 15.0) …ho are zuph, the probability of xevu is 55%. For those who are not zuph, the probability of rixq is 30%. For those who are zuph, the probability of rixq is 11%. | Will rixq increase the chance of xevu?
- (frontdoor, nonsense5, correlation, act 14.6) …effect on kwox. Qwiu is unobserved. The overall probability of jyka is 88%. The probability of not jyka and kwox is 8%. The probability of jyka and kwox is 65%. | Is the chance of kwox smaller when observing jyka?
- (frontdoor, nonsense5, correlation, act 13.9) …ffect on kwox. Qwiu is unobserved. The overall probability of jyka is 82%. The probability of not jyka and kwox is 14%. The probability of jyka and kwox is 50%. | Is the chance of kwox smaller when observing jyka?
- (frontdoor, nonsense5, correlation, act 13.9) …ffect on kwox. Qwiu is unobserved. The overall probability of jyka is 82%. The probability of not jyka and kwox is 14%. The probability of jyka and kwox is 50%. | Is the chance of kwox larger when observing jyka?
- (frontdoor, nonsense3, correlation, act 13.3) …ffect on xevu. Cwoi is unobserved. The overall probability of yomx is 43%. The probability of not yomx and xevu is 30%. The probability of yomx and xevu is 31%. | Is the chance of xevu smaller when observing yomx?

**Feature 6267** — AUROC discovery 0.71, held-out stories 0.64; fires on 43% of mediation vs 11% of others; top-8 purity 0%

- (collision, nonsense7, exp_away, act 16.7) …ty of uvzi is 65%. For those who are zuph and are not wibl, the probability of uvzi is 10%. For those who are zuph and are wibl, the probability of uvzi is 90%. | If we look at those who are wibl, does the chance of uvzi increase when zuph?
- (collision, nonsense6, exp_away, act 16.5) …ty of muvq is 98%. For those who are yupt and are not zupj, the probability of muvq is 90%. For those who are yupt and are zupj, the probability of muvq is 87%. | If we look at those who are zupj, does the chance of muvq decrease when yupt?
- (collision, nonsense5, exp_away, act 15.4) …ty of kwox is 12%. For those who are jyka and are not yupt, the probability of kwox is 13%. For those who are jyka and are yupt, the probability of kwox is 64%. | If we look at those who are yupt, does the chance of kwox decrease when jyka?
- (collision, nonsense2, exp_away, act 12.9) …ty of gyzp is 80%. For those who are xevo and are not tijv, the probability of gyzp is 52%. For those who are xevo and are tijv, the probability of gyzp is 32%. | If we look at those who are tijv, does the chance of gyzp increase when xevo?
- (collision, nonsense8, exp_away, act 12.9) …y of rukz is 99%. For those who are pexu and are not hwax, the probability of rukz is 92%. For those who are pexu and are hwax, the probability of rukz is 100%. | If we look at those who are hwax, does the chance of rukz decrease when pexu?

**Feature 5988** — AUROC discovery 0.70, held-out stories 0.62; fires on 33% of mediation vs 3% of others; top-8 purity 100%

- (mediation, nonsense5, det-counterfactual, act 17.0) … or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka and yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?
- (mediation, nonsense1, det-counterfactual, act 15.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is not kwoz if kwox instead of not kwox?
- (mediation, nonsense1, det-counterfactual, act 15.2) … or causal relationships: Kwox has a direct effect on swoq and kwoz. Swoq has a direct effect on kwoz. We know that kwox causes swoq. kwox and swoq causes kwoz. | Would an individual is kwoz if kwox instead of not kwox?
- (mediation, nonsense6, det-counterfactual, act 14.9) … or causal relationships: Yupt has a direct effect on zupj and muvq. Zupj has a direct effect on muvq. We know that yupt causes zupj. yupt and zupj causes muvq. | Would an individual is muvq if not yupt instead of yupt?
- (mediation, nonsense5, det-counterfactual, act 14.9) …s or causal relationships: Jyka has a direct effect on yupt and kwox. Yupt has a direct effect on kwox. We know that jyka causes yupt. jyka or yupt causes kwox. | Would an individual is kwox if not jyka instead of jyka?

**Feature 3198** — AUROC discovery 0.70, held-out stories 0.57; fires on 89% of mediation vs 59% of others; top-8 purity 12%

- (diamondcut, nonsense9, correlation, act 26.3) …ory has a direct effect on glimx. The overall probability of zuph is 55%. The probability of not zuph and glimx is 16%. The probability of zuph and glimx is 8%. | Is the chance of glimx smaller when observing zuph?
- (frontdoor, nonsense9, correlation, act 25.9) …ect on glimx. Zory is unobserved. The overall probability of zuph is 11%. The probability of not zuph and glimx is 33%. The probability of zuph and glimx is 4%. | Is the chance of glimx larger when observing zuph?
- (diamondcut, nonsense1, correlation, act 25.5) …. Muvy has a direct effect on kwoz. The overall probability of kwox is 66%. The probability of not kwox and kwoz is 27%. The probability of kwox and kwoz is 9%. | Is the chance of kwoz smaller when observing kwox?
- (arrowhead, nonsense1, correlation, act 24.9) …effect on kwoz. Muvy is unobserved. The overall probability of kwox is 12%. The probability of not kwox and kwoz is 22%. The probability of kwox and kwoz is 8%. | Is the chance of kwoz smaller when observing kwox?
- (mediation, nonsense3, correlation, act 24.7) …. Gwet has a direct effect on xevu. The overall probability of yomx is 65%. The probability of not yomx and xevu is 11%. The probability of yomx and xevu is 6%. | Is the chance of xevu larger when observing yomx?

**Feature 11433** — AUROC discovery 0.68, held-out stories 0.59; fires on 47% of mediation vs 28% of others; top-8 purity 38%

- (diamond, nonsense9, det-counterfactual, act 13.6) …fect on zory and jyka. Jyka has a direct effect on glimx. Zory has a direct effect on glimx. We know that zuph causes jyka and zory. jyka and zory causes glimx. | Would an individual is glimx if zuph instead of not zuph?
- (mediation, nonsense6, det-counterfactual, act 12.3) … or causal relationships: Yupt has a direct effect on zupj and muvq. Zupj has a direct effect on muvq. We know that yupt causes zupj. yupt and zupj causes muvq. | Would an individual is muvq if not yupt instead of yupt?
- (mediation, nonsense8, det-counterfactual, act 12.2) … or causal relationships: Pexu has a direct effect on hwax and rukz. Hwax has a direct effect on rukz. We know that pexu causes hwax. pexu and hwax causes rukz. | Would an individual is not rukz if pexu instead of not pexu?
- (fork, nonsense8, correlation, act 11.8) …z. Hwax has a direct effect on rukz. The overall probability of pexu is 0%. The probability of not pexu and rukz is 44%. The probability of pexu and rukz is 0%. | Is the chance of rukz smaller when observing pexu?
- (diamond, nonsense3, det-counterfactual, act 11.6) …ect on cwoi and gwet. Gwet has a direct effect on xevu. Cwoi has a direct effect on xevu. We know that yomx causes gwet and not cwoi. gwet and cwoi causes xevu. | Would an individual is xevu if not yomx instead of yomx?