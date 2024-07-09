//naive bayes classifier in C++
// necessary libraries used

#include <iostream>
using namespace std;
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <cctype>
#include<limits>
#include<iomanip>
#include<cctype>
#include<map>
#include<unordered_map>
#include<fstream>
#include<sstream>




//read the expense from the directory passed as a parameter

vector<string> generate_expenses(string filepath) {
    fstream dataset;
    dataset.open(filepath);
    vector<string> expenses;
    vector<string>expensesClasses;
    string line, expense, expenseClass;
    getline(dataset, line);
    while (getline(dataset, line)) {
        stringstream reader(line);
        getline(reader, expense, ',');
        getline(reader, expenseClass, ',');
        expenses.push_back(expense);
        expensesClasses.push_back(expenseClass);
    }
    dataset.close();
    return expenses;
}
//read the CSV file from the directory parameter
vector<string> generate_expense_classes(string filepath) {
    fstream dataset;
    dataset.open(filepath);
    vector<string> expenses;
    vector<string>expensesClasses;
    string line, expense, expenseClass;
    getline(dataset, line);
    while (getline(dataset, line)) {
        stringstream reader(line);
        getline(reader, expense, ',');
        getline(reader, expenseClass, ',');
        expenses.push_back(expense);
        expensesClasses.push_back(expenseClass);
    }
    dataset.close();
    return expensesClasses;
}
//tokenize each word into a two dimenional array
// first dimenion represents a complete expense, but the second dimension represents words in each expenses(basically tokenized verion of the expense)
vector<vector <string>> convert_array_into_word_2d(vector<vector <string>> vector_2d) {
    vector< vector<string>> elements_array_2d;
    elements_array_2d.resize(vector_2d.size());
    for (int i = 0; i < vector_2d.size(); i++) {
        string current_element = vector_2d[i][0];
        string current_token;
        int counter = 0;
        for (int j = 0; j < vector_2d[i][0].size(); j++) {
            if (current_element[j] != ' ') {
                current_token += current_element[j];
                counter++;
            }
            else {
                elements_array_2d[i].push_back(current_token);
                current_token.clear();
            }
        }if (!current_token.empty()) {
            elements_array_2d[i].push_back(current_token);
        }
    }
    return elements_array_2d;
}

//first preprocessor
vector<vector <string>> preprocess_text_data(vector<vector <string>> elements_array_2d) {
    vector<string>stop_words{ "Paid", "Invested", "on", "On", "for", "For", "Bought", "bought","paid" ,"and", "in", "expense" };
    //string stop_words[11] = { "Paid","Invested","on","On","for","For","Bought","bought","invested","in","expense" };
    for (int i = 0; i < elements_array_2d.size(); i++) {
        for (int j = 0; j < elements_array_2d[i].size(); j++) {
            for (int k = 0; k < stop_words.size(); k++) {
                if (stop_words[k] == elements_array_2d[i][j]) {
                    elements_array_2d[i].erase(elements_array_2d[i].begin() + j);
                }
            }
        }
    }
    return elements_array_2d;
}
//preprocess the text data again(there were some errors such as "vector subscript out of range" if I modified the previous preprocessing function by adding new words)
vector<vector <string>> preprocess_text_data_2(vector<vector <string>> elements_array_2d) {
    vector<string>stop_words{ "Purchased", "invested","Hired",
        "a","in","an","of","and",
        "Renovated", "Organized", "Implemented", "Renewed",
        "Hosted", "bought","Paid" ,"invested", "in", "expense","to","the","for" };
    //string stop_words[11] = { "Paid","Invested","on","On","for","For","Bought","bought","invested","in","expense" };
    for (int i = 0; i < elements_array_2d.size(); i++) {
        for (int j = 0; j < elements_array_2d[i].size(); j++) {
            for (int k = 0; k < stop_words.size(); k++) {
                if (stop_words[k] == elements_array_2d[i][j]) {
                    elements_array_2d[i].erase(elements_array_2d[i].begin() + j);
                }
            }
        }
    }
    return elements_array_2d;
}


//generate word frequency into a hashmap or a format which are dictionaries in Python
unordered_map<string, unordered_map<string, double>> generate_word_frequencies(vector< vector<string>> elements_array_2d, vector<string>expensesClasses) {
    unordered_map<string, unordered_map<string, double>> words_frequency_map;
    int admin = 0, dist = 0, fin = 0;

    for (size_t i = 0; i < elements_array_2d.size(); i++) {
        string category = expensesClasses[i];
        for (size_t j = 0; j < elements_array_2d[i].size(); j++) {
            string word = elements_array_2d[i][j];
            words_frequency_map[word]["Administrative expenses"];
            words_frequency_map[word]["Distribution cost"];
            words_frequency_map[word]["Finance cost"];
            words_frequency_map[word][category]++;
        }
        for (size_t k = 0; k < elements_array_2d[i].size(); k++) {
            string word = elements_array_2d[i][k];
            if (words_frequency_map[word]["Administrative expenses"] == 0) {
                words_frequency_map[word]["Administrative expenses"]++;
            }
            if (words_frequency_map[word]["Distribution cost"] == 0) {
                words_frequency_map[word]["Distribution cost"]++;
            }
            if (words_frequency_map[word]["Finance cost"] == 0) {
                words_frequency_map[word]["Finance cost"]++;
            }
        }
    }
    return words_frequency_map;
}

//calculate word count for every expense class, for example administrative expenses, may have X number of words, distribution costs may have Y number of words, and finance costs may have Z number of words
double calculate_word_count(unordered_map<string, unordered_map<string, double>> words_frequency_map, string cls) {
    double sum = 0;
    for (const auto& pair : words_frequency_map) {
        string word = pair.first;
        sum += words_frequency_map[word][cls];
    }
    return sum;
}

//calculate likelihood for each word
unordered_map<string, unordered_map<string, double>> calculate_likelihood(unordered_map<string, unordered_map<string, double>> words_frequency_map, string cls, double class_word_count) {
    unordered_map<string, unordered_map<string, double>> likelihood;
    for (const auto& pair : words_frequency_map) {
        string word = pair.first;
        likelihood[word][cls] = words_frequency_map[word][cls] / class_word_count;
    }
    return likelihood;
}

//calculate prior for each word
unordered_map<string, unordered_map<string, double>> calculate_P_word(unordered_map<string, unordered_map<string, double>> frequency, string cls, double prior) {
    unordered_map<string, unordered_map<string, double>> P_Of_word;
    for (const auto& pair : frequency) {
        string word = pair.first;
        P_Of_word[word][cls] += frequency[word][cls] * prior;
    }
    return P_Of_word;
}



//function to classify new expenses, based on the previously calculated posterior and priors for each word and expense
string classify_new_expenses(vector<vector<string>> my_new_2d_vector,
    unordered_map<string, unordered_map<string, double>> admin_likelihood,
    unordered_map<string, unordered_map<string, double>> dist_likelihood,
    unordered_map<string, unordered_map<string, double>> fin_likelihood,
    double prior_admin, double prior_dist, double prior_fin) {
    string result = "";
    double admin_prob = 1.0, dist_prob = 1.0, fin_prob = 1.0;
    for (int i = 0; i < my_new_2d_vector.size(); i++) {
        for (int j = 0; j < my_new_2d_vector[i].size(); j++) {
            const string word = my_new_2d_vector[i][j];
            if (admin_likelihood[word]["Administrative expenses"] != 0) {
                admin_prob *= admin_likelihood[word]["Administrative expenses"];
            }
            if (dist_likelihood[word]["Distribution cost"] != 0) {
                dist_prob *= dist_likelihood[word]["Distribution cost"];
            }
            if (fin_likelihood[word]["Finance cost"] != 0) {
                fin_prob *= fin_likelihood[word]["Finance cost"];
            }
        }
    }
    admin_prob *= prior_admin;
    dist_prob *= prior_dist;
    fin_prob *= prior_fin;
    if (admin_prob > dist_prob && admin_prob > fin_prob) {
        result = "Administrative expense";
    }
    else if (dist_prob > admin_prob && dist_prob > fin_prob) {
        result = "Distribution cost";
    }
    else {
        result = "Finance cost";
    }
    return result;
}
//function with all the functions of naive bayes integrated
void naive_bayes(vector<string>expenses, vector<string>expensesClasses) {

    vector<vector<string>> vector_2d;
    vector_2d.resize(expenses.size());
    for (int i = 0; i < vector_2d.size(); i++) {
        vector_2d[i].resize(1);
        vector_2d[i][0] = expenses[i];
    }
    /*for (int i = 0; i < expenses.size(); i++) {
     cout << expenses[i] << endl;
     }*/
    vector< vector<string>> elements_array_2d = convert_array_into_word_2d(vector_2d);
    elements_array_2d = preprocess_text_data(elements_array_2d);
    elements_array_2d = preprocess_text_data_2(elements_array_2d);
    unordered_map<string, unordered_map<string, double>> words_frequency_map = generate_word_frequencies(elements_array_2d, expensesClasses);

    cout << "number of expenses the model is trained on: " << expenses.size() << endl;
    double prob_admin = 1, prob_dist = 1, prob_fin = 1, total_admin = 0, total_dist = 0, total_fin = 0;
    for (int i = 0; i < expensesClasses.size(); i++) {
        if (expensesClasses[i] == "Administrative expenses") {
            total_admin++;
        }
        if (expensesClasses[i] == "Distribution cost") {
            total_dist++;
        }
        if (expensesClasses[i] == "Finance cost") {
            total_fin++;
        }
    }
    double total_count = expenses.size();
    double prior_admin = total_admin / total_count;
    double prior_dist = total_dist / total_count;
    double prior_fin = total_fin / total_count;
    double admin_word_count = calculate_word_count(words_frequency_map, "Administrative expenses");
    double dist_word_count = calculate_word_count(words_frequency_map, "Distribution cost");
    double fin_word_count = calculate_word_count(words_frequency_map, "Finance cost");
    cout << "Number of words in administrative expenses: " << admin_word_count << "\n" << "Number of words in distribution cost: " << dist_word_count << "\n" <<
        "Number of words in finance cost: " << fin_word_count << endl;
    unordered_map<string, unordered_map<string, double>> admin_likelihood = calculate_likelihood(words_frequency_map, "Administrative expenses", admin_word_count);
    unordered_map<string, unordered_map<string, double>> dist_likelihood = calculate_likelihood(words_frequency_map, "Distribution cost", dist_word_count);
    unordered_map<string, unordered_map<string, double>> fin_likelihood = calculate_likelihood(words_frequency_map, "Finance cost", fin_word_count);
    vector<string> Test_expenses = {
        "Office supplies purchase","Employee salaries",
        "Rent for office space","Utility bills payment",
        "Legal consultation fees","Accounting software subscription",
        "Employee training program costs","Travel expenses for business meetings",
        "Marketing and advertising expenses","Office equipment maintenance",
        "Insurance premiums for business property","Office cleaning services",
        "IT support services","Website hosting fees",
        "Office furniture purchase","Employee health insurance premiums",
        "Professional development courses fees","Tax preparation and filing fees",
        "Employee retirement plan contributions","Internet and phone service charges",

        "Freight charges for shipping goods to retailers","Warehousing fees for storing inventory",
        "Transportation costs for delivering products to customers","Packaging materials for protecting items during shipping",
        "Distribution center rent for storing and managing inventory","Delivery truck maintenance and repairs",
        "Shipping insurance premiums to cover potential losses","Inventory management software subscription fees",
        "Packaging design and labeling costs","Third-party logistics services for order fulfillment",
        "Customs duties and import taxes for international shipments","Warehouse security system installation and monitoring fees",
        "Route optimization software licensing fees","Distribution network expansion expenses",
        "Last-mile delivery expenses for reaching customers' doorsteps","Inventory tracking and management system implementation costs",
        "Cross-docking facility rent and operations","Shipping container purchase or rental fees",
        "Distribution center staffing and training expenses","Forklift purchase or lease payments",


        "Interest payments on loans and lines of credit",
        "Bank loan origination fees",
        "Bond issuance expenses, including underwriting and legal fees",
        "Debt service fees for managing outstanding loans",
        "Credit card processing fees for customer transactions",
        "Financial advisory and consulting services fees",
        "Treasury management software subscription costs",
        "Investment brokerage commissions for buying and selling securities",
        "Financial audit fees for annual compliance checks",
        "Insurance premiums for business interruption coverage",
        "Tax preparation and filing fees",
        "Hedge fund management fees",
        "Loan default insurance premiums",
        "Foreign exchange transaction fees",
        "Securities registration and regulatory compliance costs",
        "Asset valuation and appraisal fees",
        "Financial statement audit fees",
        "Pension fund administration fees",
        "Investment portfolio management fees",
        "Legal expenses related to financial litigation or regulatory issues",
        "paid interest to the bank"
    };
    admin_likelihood = calculate_P_word(admin_likelihood, "Administrative expenses", prior_admin);
    dist_likelihood = calculate_P_word(dist_likelihood, "Distribution cost", prior_dist);
    fin_likelihood = calculate_P_word(fin_likelihood, "Finance cost", prior_fin);
    vector<vector<string>> my_2d_vector;

    cout << "Please enter the new expense to classify " << endl;
    int i = 0;
    /*cout << "Following expense are truly administrative expenses " << endl;
     for (int i = 0; i < Test_expenses.size(); i++) {
     string new_expense = Test_expenses[i];
     vector<string> inner_vector = { new_expense };
     my_2d_vector.push_back(inner_vector);
     my_2d_vector = convert_array_into_word_2d(my_2d_vector);
     my_2d_vector = preprocess_text_data(my_2d_vector);
     my_2d_vector = preprocess_text_data_2(my_2d_vector);
     string result = classify_new_expenses(my_2d_vector, admin_likelihood, dist_likelihood, fin_likelihood, prior_admin, prior_dist, prior_fin);
     cout << "The model classifies the expense \n" << Test_expenses[i] << " as \n" << result << endl;
     my_2d_vector.clear();
     cout << "\n\n\n" << endl;
     }*/
    char c;
    cout << "Want to enter an expense ?(y/n) " << endl;
    cin >> c;
    while (true) {
        cin.ignore();
        if (c == 'y') {
            string new_expense;
            cout << "Enter the expense " << endl;
            getline(cin, new_expense);
            vector<string> inner_vector = { new_expense };
            my_2d_vector.push_back(inner_vector);
            my_2d_vector = convert_array_into_word_2d(my_2d_vector);
            my_2d_vector = preprocess_text_data(my_2d_vector);
            my_2d_vector = preprocess_text_data_2(my_2d_vector);
            cout << "Expense after preprocessing " << endl;
            for (int i = 0; i < my_2d_vector.size(); i++) {
                for (int j = 0; j < my_2d_vector[i].size(); j++) {
                    cout << my_2d_vector[i][j] << " ";
                }
            }
            cout << endl;
            string result = classify_new_expenses(my_2d_vector, admin_likelihood, dist_likelihood, fin_likelihood, prior_admin, prior_dist, prior_fin);
            cout << result << endl;
            cout << "Another expense? " << endl;
            my_2d_vector.clear();
        }
        else {
            cout << "Please enter a correct input " << endl;
        }
        cin >> c;
    }
}


//Musab's classifier in one function
void main_console_classifier() {
            string filepath = "Your path to the CSV file";
            vector<string>expenses = generate_expenses(filepath);
            vector<string>expenseClasses = generate_expense_classes(filepath);
            naive_bayes(expenses, expenseClasses);

    
}
// call the main_console_classifier in the main function, you can write your own main function as per your choice
// Kindly look at my work and kindly give me your critique after looking at it
