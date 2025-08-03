#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

// RNG for dice rolls
static std::mt19937 rng(std::random_device{}());

// d(k): roll a k-sided die, returning 1..k
int d(int k) {
    std::uniform_int_distribution<int> dist(1, k);
    return dist(rng);
}

int play_once() {
    int player1 = 0, player2 = 0;
    int money1 = 17, money2 = 17;
    std::vector<int> price = {3, 4, 5, 4};
    std::vector<int> rent  = {1, 1, 2, 2};
    std::vector<int> owners(4, 0);

    for (int t = 0; t < 5; ++t) {
        // -- Player 1 turn --
        if (money1 > 0 && money2 > 0) {
            player1 = (player1 + d(6)) % 12;
            if (player1 % 3 == 2) {
                int card = d(2);
                money1 -= (card == 1 ? 1 : 5);
            }
            if (player1 % 3 == 1) {
                int prop = player1 / 3;
                if (owners[prop] == 0 && money1 >= price[prop]) {
                    owners[prop] += 1;
                    money1 -= price[prop];
                }
                if (owners[prop] == 2) {
                    money1 -= rent[prop];
                    money2 += rent[prop];
                }
            }
            money1 = std::max(money1, 0);
        }

        // -- Player 2 turn --
        if (money1 > 0 && money2 > 0) {
            player2 = (player2 + d(6)) % 12;
            if (player2 % 3 == 2) {
                int card = d(2);
                money2 -= (card == 1 ? 1 : 5);
            }
            if (player2 % 3 == 1) {
                int prop = player2 / 3;
                if (owners[prop] == 0 && money2 >= price[prop]) {
                    owners[prop] += 2;
                    money2 -= price[prop];
                }
                if (owners[prop] == 1) {
                    money2 -= rent[prop];
                    money1 += rent[prop];
                }
            }
            money2 = std::max(money2, 0);
        }
    }

    return money1;
}

int main() {
    const int N = 100000000;  // number of trials
    std::map<int, int> counts;

    // Run trials
    for (int i = 0; i < N; ++i) {
        int result = play_once();
        counts[result]++;
    }

    // Print distribution
    std::cout << "money1\tcount\tprobability\n";
    std::cout << "--------------------------------\n";
    std::cout << std::fixed << std::setprecision(5);
    for (auto [money, cnt] : counts) {
        double prob = double(cnt) / N;
        std::cout << std::setw(6) << money << '\t'
                  << std::setw(7) << cnt  << '\t'
                  << prob << '\n';
    }

    return 0;
}
