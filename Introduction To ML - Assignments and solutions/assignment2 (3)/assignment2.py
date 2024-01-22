#################################
# Your name: Mira Zilberstein
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.sort(np.random.uniform(low=0.0, high=1.0, size=m))
        ys = [np.random.choice([1, 0], p=[0.8, 0.2] if (xi <= 0.2 or (0.4 <= xi <= 0.6) or xi >= 0.8) else [0.1, 0.9])
              for xi in xs]
        return [xs, ys]

    #given a list of intervals,I, calculates the true error e_p(h_I)
    def true_error(self,intervals):
        inter_len = 0 #sum of integrals lengths
        inter_len_a = 0 #sum of integrals lengths that in A= [0,0.2]u[0.4,0.6]u[0.8,1]
        for interval in intervals:
            inter_len += interval[1]-interval[0]
            if interval[0] <= 0.2:
                inter_len_a += min(interval[1], 0.2) - max(0, interval[0])
            if interval[0] <= 0.6 and interval[1] >= 0.4:
                inter_len_a += min(interval[1], 0.6) - max(0.4, interval[0])
            if interval[1] >= 0.8:
                inter_len_a += min(interval[1], 1) - max(0.8, interval[0])
        non_inter_len_a = 0.6 - inter_len_a #sum of lengths that in A= [0,0.2]u[0.4,0.6]u[0.8,1] but not in integrals
        answer = (inter_len_a*0.2 + (inter_len - inter_len_a)*0.9) + (non_inter_len_a*0.8 + ((1 - inter_len) - non_inter_len_a)*0.1)
        return answer

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        results_ep = []
        results_es = []
        for n in range(m_first, m_last+1, step):
            es_avg = 0;
            ep_avg = 0;
            for T in range(100):
                xs, ys = self.sample_from_D(n)
                interval = intervals.find_best_interval(xs, ys, k)
                es_avg += interval[1]/n
                ep_avg += self.true_error(interval[0])
            es_avg /= 100
            ep_avg /= 100
            results_ep.append(ep_avg)
            results_es.append(es_avg)

        plt.plot(range(m_first, m_last+1, step),
                 results_ep, label="average e_p")
        plt.plot(range(m_first, m_last + 1, step),
                 results_es, label="average e_s")
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('average e_p/e_s')
        plt.show()
        return [results_es,results_ep]

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        results_ep = []
        results_es = []
        xs, ys = self.sample_from_D(m)
        min_k = 1
        min_e_s_error = 1
        for k in range(k_first, k_last + 1, step):
            interval = intervals.find_best_interval(xs, ys, k)
            curr_es = interval[1] / m
            if(min_e_s_error >= curr_es):
                min_e_s_error = curr_es
                min_k = k
            results_es.append(curr_es)
            results_ep.append(self.true_error(interval[0]))

        plt.plot(range(k_first, k_last + 1, step),
                 results_ep, label="e_p")
        plt.plot(range(k_first, k_last + 1, step),
                 results_es, label="e_s")
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('e_p/e_s')
        plt.show()
        return min_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        results_ep = []
        results_es = []
        results_penalty = []
        results_penalty_plus_es = []
        xs, ys = self.sample_from_D(m)
        SRM_k = 1
        min_SRM_error = 1
        for k in range(k_first, k_last + 1, step):
            curr_penalty = 2*np.sqrt((2*k+np.log(2/0.1))/m)
            results_penalty.append(curr_penalty)
            interval = intervals.find_best_interval(xs, ys, k)
            curr_es = interval[1] / m
            results_es.append(curr_es)
            results_ep.append(self.true_error(interval[0]))
            curr_penalty_plus_es = curr_es + curr_penalty
            results_penalty_plus_es.append(curr_penalty_plus_es)
            if (min_SRM_error >= curr_penalty_plus_es):
                min_SRM_error = curr_penalty_plus_es
                SRM_k = k
        plt.plot(range(k_first, k_last + 1, step),
                 results_ep, label="e_p")
        plt.plot(range(k_first, k_last + 1, step),
                 results_es, label="e_s")
        plt.plot(range(k_first, k_last + 1, step),
                 results_penalty, label="penalty")
        plt.plot(range(k_first, k_last + 1, step),
                 results_penalty_plus_es, label="penalty + e_s")
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('e_p / e_s / penalty / penalty+e_s')
        plt.show()
        return SRM_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        training_size = int(m*0.8)
        holdout_size = int(m*0.2)
        x_t, y_t = self.sample_from_D(training_size)
        x_ho, y_ho = self.sample_from_D(holdout_size)
        min_e_s = 1
        holdout_validation_k = 1
        for k in range(1, 11):
            intervals_ERM = intervals.find_best_interval(x_t, y_t, k)[0]
            curr_e_s = 0
            for i in range(holdout_size): # calculate curr e_s
                x_prediction = 0
                for interval in intervals_ERM:
                    if interval[0] <= x_ho[i] <= interval[1]:
                        x_prediction = 1
                        break
                    if interval[0] > x_ho[i]:
                        break
                if(x_prediction != y_ho[i]):
                    curr_e_s += 1
            curr_e_s = curr_e_s/(holdout_size)

            if(curr_e_s < min_e_s):
                holdout_validation_k = k
                min_e_s = curr_e_s

        return holdout_validation_k


    #################################
    # Place for additional methods
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
