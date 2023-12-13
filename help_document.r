options(digits=11)
# Help document 
#######################################################################################################################

#Distributions

# Uniform distribution
    # A Uniform Distribution is a type of probability distribution in which all outcomes are equally likely. It’s graph is a rectangle,
    # flat with constant probability.
    # runif(samples, min, max)

# Normal distribution (z)
    # dnorm(x, mean, sd): This function calculates the density function for a normal distribution with a given mean and standard deviation.
    # It returns the height of the probability density function at the value x. For example, dnorm(0, 0, 1) returns the height of the
    # standard normal distribution at 0.
    
    # pnorm(q, mean, sd): This function calculates the cumulative distribution function for a normal distribution with a given
    # mean and standard deviation. It returns the probability that a normally distributed random variable with this mean and
    # standard deviation will be less than or equal to q. For example, pnorm(1.96, 0, 1) returns the probability that a standard normal
    # variable is less than or equal to 1.96, which is approximately 0.975.
    
    # # qnorm(p, mean, sd): This function calculates the quantile function for a normal distribution with a given mean and
    # standard deviation. It returns the value of the random variable such that the probability of the variable being less
    # than or equal to that value equals the given probability p. For example, qnorm(0.975, 0, 1) returns the 97.5% quantile
    # of the standard normal distribution.

# Poisson distribution (Discrete version of Exponential Distribution)
    # dpois(x, lambda) = Poisson Density Function. Returns the probability that a certain number of events (denoted by x)
    # occur in a fixed interval of time. Given fixed avg rate of occurence (lambda). For example, if a website is known to
    # make 10 sales per hour (lambda = 10), you can use dpois() to find the probability that the site makes exactly 8
    # sales in a given hour(x=8).
    
    # ppois(q, lambda) = Poisson Cumulative Densitiy Function. It returns the probability that a certain number of events
    # or less (denoted by q) occur in a fixed interval of time. Given a fixed avg. rate of occurence (lambda).
    # For instance if a website is known to make 10 sales per hour (lambda = 10), you can use ppois() to find the
    # probability that the sitee makes 8 sales or less in a given hour (q = 8).
    
    # qpois(p, lambda): This function calculates the quantile function for a Poisson distribution with a given rate lambda.
    # It returns the largest number of events such that the probability of that many or fewer events is less than or equal to p.
    # For example, qpois(0.90, 10) returns the 90% quantile of a Poisson distribution with a rate of 10.
    
    # Rate: λ
    # 1. Identify the number of events: Count the number of events that occur in your observation.
    # This could be anything from the number of emails you receive in an hour to the number
    # of sales a website makes per day.
    # 2. Identify the interval: Determine the length of the interval during which the events occur.
    # This could be an hour, a day, a square meter, etc.
    # 3. Calculate the rate: Divide the number of events by the length of the interval.
    # This gives you the average rate of occurrence, λ123.
    
    # Mean: μ = λ
    # Variance: α^2 = λ

# Exponential Distribution (Continuous version of Poisson Distribution)
    # Can be used to describe waiting time between poisson events.
    
    # dexp(x, rate) = Exponential Density Function. Returns the probability density at a certain time point (denoted by x)
    # given a fixed average rate of occurrence (rate). For example, if a website is known to make 10 sales per hour (rate = 10),
    # you can use dexp() to find the density of making a sale at a specific time point within an hour.
    
    # pexp(q, rate) = Exponential Cumulative Density Function. It returns the probability that a certain
    # amount of time or less (denoted by q) will pass before the next event, given a fixed average rate
    # of occurrence (rate). For instance, if a website is known to make 10 sales per hour (rate = 10),
    # you can use pexp() to find the probability that the next sale will occur within a certain amount of time.
    
    # qexp(p, rate): This function calculates the quantile function for an exponential distribution with a given rate. It returns
    # the waiting time such that the probability of waiting less than or equal to that time equals p. For example, qexp(0.90, 10) returns
    # the 90% quantile of an exponential distribution with a rate of 10.
    
    # Rate: λ = 1/mean.
    # Mean: μ = 1/λ
    # Variance: σ^2 = 1/λ^2

# Binomial distribution (with replacement)
    # “at least” two, so it is one minus the probability of ’less
    # than or equal to 1’
    # Binary outcome: Either success or failure.
    # Fixed number of trials.
    # Independent trials, each trial is independent meaning one outcome does not affect another.
    # Constant probability of success
    
    # dbinom(x, size, prob): This function calculates the probability mass function (PMF) for a binomial distribution
    # with a given number of trials (size) and probability of success (prob). It returns the probability of getting
    # exactly x successes in size trials. The PMF of x is given by the formula
    # x = number of successes, size = number of trials, prob = probability of success.
    
    # pbinom(x, size, prob): This function calculates the cumulative distribution function (CDF) for a binomial
    # distribution with a given number of trials (size) and probability of success (prob). It returns the
    # probability of getting x or fewer successes in size trials. The CDF is given by the formula
    
    # qbinom(p, size, prob): This function calculates the quantile function for a binomial distribution with a given number
    # of trials size and probability of success prob. It returns the largest number of successes such that the probability of
    # that many or fewer successes is less than or equal to p. For example, qbinom(0.90, 100, 0.5) returns the 90% quantile of
    # a binomial distribution with 100 trials and a success probability of 0.5.
    
    # Mean: μ = n * p
    # Variance: α^2 = n*p*(1-p)

# Hypergeometric distribution (Without replacement)
    # dhyper(x, a, N-a, n-x) Calculates probability mass function (PMF) and returns the probability of drawing exactly X white balls.
    # x = probability of x successes drawn (What they are asking for, for example "Whats the probability of drawing 8 white balls?").
    # a = number of white balls in the urn (successes)
    # N-a = number of black balls in the urn (failures)
    # n-x = number of balls drawn from the urn.
    
    # phyper(x, a, N-a, n-x) Calculates the cumulative distribution function (CDF) and returns the probability of drawing
    # x or fewer white balls.
    
    # qhyper(p, m, n, k): This function calculates the quantile function for a hypergeometric distribution with m white balls,
    # n black balls, and k draws. It returns the largest number of white balls such that the probability of drawing that many
    # or fewer white balls is less than or equal to p. For example, qhyper(0.90, 50, 50, 10) returns the 90% quantile of a hypergeometric
    # distribution with 50 white balls, 50 black balls, and 10 draws.
    
    # Mean: μ = n * a/N
    # Variance: α^2 = na*(N-a)(N-n)/N^2*(N-1)

# A Chi-Square Distribution (X^2, Asymetric - Probability = 0.95 for alpha = 0.05)
    # Is a type of probability distribution that is used in hypothesis testing and is based on a normally distributed random variable.
    # It is used in various statistical applications involving degrees of freedom.
    
    # pchisq(q, df): This function calculates the cumulative distribution function for a chi-square distribution with a given
    # degrees of freedom. It returns the probability that a chi-square distributed random variable with this degrees of freedom will be less than or equal to q. 
    # For example, pchisq(3.84, 1) returns the probability that a chi-square variable with 1 degree of freedom is less than or equal to 3.84, which is approximately 0.95.
    
    # qchisq(p, df): This function calculates the quantile function for a chi-square distribution with a given degrees of freedom.
    # It returns the value of the random variable such that the probability of the variable being less than or equal to that value equals the given probability p. 
    # For example, qchisq(0.95, 1) returns the 95% quantile of the chi-square distribution with 1 degree of freedom.
    
    # Mean: μ = df
    # Variance: α^2 = 2*df
    # df = (r−1)(c−1) Where r = rows, c = columns.

#######################################################################################################################

# Hypothesis Testing
    # With the null hypothesis H0 = mean = 0
    # If confidence interval does NOT contain 0 we reject the null hypothesis!
    # If confidence interval contains 0 we we fail to reject the null hypothesis. (Cannot conclude that one is better than the other)
    # If t value > t critical value we reject the null hypothesis based on alpha. (No p-value needed then).
    # If p value < alpha value we reject the H0 hypothesis that there is no difference.

# One sample t-test
    # Sample size formula (how many observations needed): n <- (sd*(qnorm(1-(sig/2))+qnorm(1-beta))/ME)^2
    # sig = significance (0.05 for example)
    # sd = standard deviation.
    # power = statistical power (1 - beta). Likelihood of a significance test detecting
    # an effect when there actually is one. It is sometimes also called sensitivity. If power is 45, then there is a 65% chance of making a type 2 error.
    # beta = typically 0.2 (probability of committing a Type II error which is failing to reject the null hypothesis when its actually false.
    # ME = difference in mean.
    # Degrees of freedom: n - 1 where n is the sample size.
    
    # t.test(x) H0: mean = 0.
    # If you want to test a different hypothesis, for example that mu = 10000.
    # data <- c(12, 13, 14, 15, 16)  # your data
    # mu <- 15  # known population mean
    # t_test_result <- t.test(data, mu = 10000)

# Calculate confidence interval using sample mean and t-statistic:
    # SEM = Standard error
    # t-statistic = sample_mu - hypothesis_mu / SEM
    # hypothesis_mu = 0, since we are testing if the mean is different from zero
    # This can be rearranged like: 
    # SEM = sample_mu / t-statistic
    # Now we can calculate the standard error to be used in calculating the confidence interval
    # CI = sample_mu +- t1-a/2 * SEM
    # in R: sample_mu + c(-1,1) * qt(1-alpha, df=df) * SEM

# Paired t-test = one-sample analysis
    # Degrees of freedom: n-1 where n is the number of pairs.
    # Test used to compare a single population before and after som experiment or at two
    # different points in time. For example, measuring student performance on a test before
    # and after being taught the material.
    # pre_treatment <- c(1,2,3,4,5)
    # post_treatment <- c(2,3,4,5,6)
    # result <- t.test(pre_treatment, post_treatment, paired = TRUE)

# Two sample t-test 
    # Welch two sample t-test:
    # Degrees of freedom when the variances of the two populations are assumed to be the same: (n1 + n2 - 2)
    # Degrees of freedom otherwise = df <- ((sd1^2/n1 + sd2^2/n2)^2) / ((sd1^4/(n1^2*(n1-1)) + sd2^4/(n2^2*(n2-1))))
    # Compare means of two different groups, avg height of men differs from avg height of women.
    # t.test(x2, x1)

    # Pooled two sample t-test:
    # This requires that the two populations has equal true variance.
    # t.test(x1, x2, var.equal=TRUE) # var.equal == The variance is equal... 

# Power t-test
    # power.t.test(delta = delta, sd = sd, power = power, sig.level = sig.level, type = type)
    # n = number of observations for 1 group
    # delta = true difference in mean.
    # sd = standard deviation
    # sig.level = significance level
    # power = power of test
    # type = "one.sample", "two.sample", "paired" (can be disregarded)

# Pearsons chisquare test (X^2):
    # Used in hypothesis testing to compare observed data with data we would expect to obtain according to
    # a specific hypothesis. 
    # 1. Goodness of fit: Observed frequencies of an event match the expected frequencies. Example: die roll 60
    # times expect each face to come up 10 times.
    # 2. Test of independence: Two categorical variables and you want to see if they are related. Relation between
    # famle/male and see if they dislike / like a product.
    # 3. Test of homogenity: Compare the distribution of categorical variables in more than one population. 
    # Example: difference in the distribution of blood types in two different countries.
    # R example:
    # unemployed <- c(10, 15, ...)
    # employed <- c(24, 32, ...)
    # data <- rbind(unemployed, employed)
    # chisq.test(data, correct = FALSE)
    # p-value: 1-pchiq(x-squared, df)
    # where df = (r−1)(c−1) Where r = rows, c = columns.

# P("Type I error") = P("Rejection of H0 when H0 is true") = alpha = (usually) 0.05
# The type I error is equal to the significance level
# p value for test statistic
# 2*(1-pt(tstatistic, df))

#######################################################################################################################

# Linnear Regression Model
    # Simple model: Y = B0 + B1*x1 + ... + e (residual)
    
    # The coefficient of determination (R^2 percent, amount of variance explained by the model):
    # Is a measure that indicates how well a statsitical model predicts an outcome. Its a number between 0 and 1.
    # How to interpret it:
    # 0: The model does not predict the outcome
    # Between 1 and 0: The model partially predicts the outcoem.
    # 1: The model prefectly predicts the outcome.
    
    # The correlation coefficient (R), measures the strength and direction of the relationship between two variables.
    # It can range from -1 to 1. Here's how to interpret it:
    # Between 0 and 1: Positive correlation. When one variables changes, the other variable changes in the same direction.
    # 0: No correlation. There is no relationship between the variables.
    # Between 0 and -1: Negative correlation: When one variable changes, the other variables changes in the opposite direction.
    
    # Number of observations in the study:
    # Degrees of freedom: Numbers of data points that are free to vary after the model has been fitted, i.e. the
    # number of observations minus the number of estimated parameters.
    # Calculated: So if we have (intercept), x1, x2 we take Df for residuals + number of parameters = df + 3.
    
    # Larger t values correspond to smaller p values, indicating greater significance.
    # Very small p values correspond to significant relationship between variable x and y. 
    
    # p-value: 2*(1-pt(t-value, df)) H0 = 0.
    # Where df = the df of the attribute.
    
    # Multiple Linear Regression Model
    # df = n - (p + 1)
    # n is the total number of observations,
    # p is the number of predictor variables in the model.

#######################################################################################################################

# Simulation
    # Parametric = Simulate multiple samples from an assumed distribution.
    # Non parametric = Simulate samples from already existing observations.
    
    # Comparing differences (mean, median, standard deviation) in parametric bootstrapping:
    
    # rdist = chosen distribution to generate from
    # Use mean and sd to generate (not var for example)
    # samplesA <- replicate(nr_of_samples, rdist, mean(setA), sd(setB))
    # samplesB <- replicate(nr_of_samples, rdist, mean(setA), sd(setB))
    # simDifs  <- apply(samplesA, 2(?), mean_median_or_sd) - apply(samplesB, 2(?), mean_median_or_sd) 
    # quantile(simDifs, c(0.025, 0.975)) # 95% conf int

#######################################################################################################################

# ANOVA
    # In ANOVA we test wether the means of some groups are different from others.
    # F statistic (ANOVA): Will tell you if a single variable is statistically significant.
    # F statistic = MS(Tr)/MSE.
    # What does it represent? How much variability among the means exeedes the expected due to chance.
    # Larger F statistic, greater evidence that there is a difference between the group means. If F statistic is larger
    # there is less likelyhood that the means are due to random chance.
    # If F statistic > F critical value we reject the null hypothesis.
    # MSE estimator of the variance of the random error term. 
    # The error term represents the random variability in the data that the model doesnt capture and should not be included
    # in the model prediction.

# One way ANOVA
    # Example code:
    # y <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
    # treatments <-
    #   as.factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C",)) 
    # model <- aov(logt ~ treatments)
    # anova_table <- summary(model)
    # anova_table
    
    # p-value: 1 - pf(fstatistic, df(attribute), df(residual))
    # Total variance: SST = SSE + SS(Tr)
    # Estimated effect of treatment x1 (alphaB)
    # mean(alphaB) - mean(alphaN). Where mean(alphaN) is the mean of all observations.

    # Post Hoc Analysis:
    # No bonferroni is required if we are doing one pre-specified comparison.
    # (yhat1 - yhat2) - qt(0.975, df)*sqrt(MSE*(1/n1 + 1/n2))
    # df = n - k, SSE = variance, yhat = mean

# Two way ANOVA
    # Number of observations: k * l
    # p-value: 1 - pf(fstatistic, df(attribute), df(residual))

#######################################################################################################################

# Proportions
    # Confidence interval for 1 proportion
    
    # x = (x number out of total)
    # n = (total,  including x)
    # phat = x/n
    # phat-qnorm(1-alpha/2)*sqrt((phat*(1-phat)/n))
    # phat+qnorm(1-alpha/2)*sqrt((phat*(1-phat)/n))
    
    # Confidence interval for difference in two proportions.
    # Difference between x1 - x2:
    
    # x1 = number in group1
    # x2 = number in group2
    # n1 = total in group 1
    # n2 = total in group 2
    # alpha = 1 - alpha/2
    # phat1 = x1/n1
    # phat2 = x2/n2
    
    # (phat1 - phat2) + qnorm(alpha)*sqrt((phat1*(1-phat1)/n1)+(phat2*(1-phat2)/n2))
    # (phat1 - phat2) - qnorm(alpha)*sqrt((phat1*(1-phat1)/n1)+(phat2*(1-phat2)/n2))
    
    # Expected counts for k (Table)
    # eij = ((i'th row total) * (j'th column total))/total
    
    # Standard Error of estimated proportion
    # sd = sqrt((p(1-p))/n)
    # Where p = x/n.

# Proportion.test H0: p1 = p2
    # Define your counts of successes and trials
    # x <- c(10, 20) # replace with your counts of successes
    # n <- c(50, 100) # replace with your counts of trials
    # result <- prop.test(x, n), conf.level can be added: conf.level = 0.95

#######################################################################################################################

# Derivative

    # f <- expression(x^2 + 3*x + 2)
    # f_prime <- deriv(f, "x")
    # print(f_prime)

#######################################################################################################################

# MISCELLANOUS

# Confidence interval for mean
    # xhat <- Mean
    # s <- Standard deviation
    # n <- Number of observations
    # df <- n - 1
    # cl1 <- xhat + qt(q, df) * s/sqrt(n)
    # cl2 <- xhat - qt(q, df) * s/sqrt(n)

# IQR <- Q1 - Q3
    # The Interquartile Range (IQR) is a statistical measure used to describe the spread
    # or variability within a data set.

# Coefficient of variation: sd / mean.
    # Pooled variance (see pdf page 108) 
    # Assume that the true variance between the samples are the same.
    # n1 and n2 in the formula is the two groups, n1 = n2 = # of groups
