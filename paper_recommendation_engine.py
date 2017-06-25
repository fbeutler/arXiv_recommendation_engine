'''
Copyright 2017, Florian Beutler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

###########################################################################

This code is used to make paper recommendations for users on www.benty-fields.com.
It has two use cases:

(1) It is used to order the daily new publications for each user according to the user's interest
(2) It is used to provide paper recommendations for each user

If you have suggestions for improvements please contact benty-fields@feedback.com.
'''

from app import models, db, app, es
from sqlalchemy import or_, func, distinct, and_, not_
import datetime
import time
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt

import json
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def test():
    '''
    This function tests the recommendation engine
    '''
    with app.app_context():
        subquery = db.session.query(models.Paperclick.user_id, func.count(models.Paperclick.user_id).label('click_count')).group_by(models.Paperclick.user_id).subquery()
        users = models.User.query.join(subquery, subquery.c.user_id == models.User.id)\
                                              .order_by(subquery.c.click_count.desc())
        users = users.add_column(subquery.c.click_count).all()

        plot_x = []
        plot_y = []
        error_y = []
        plot_y2 = []
        error_y2 = []
        for i, user in enumerate(users):
            results = get_auc_through_cross_validation(user[0])

            plot_x.append(user[1])
            plot_y.append(np.mean(results['aucs']))
            error_y.append(np.std(results['aucs']))
            plot_y2.append(np.mean(results['pr']))
            error_y2.append(np.std(results['pr']))

            plt.clf()
            #y_av = moving_average(plot_y, 5)
            #plt.plot(plot_x, y_av, "r")
        
            plt.title('AUC as a function of click data objects')
            plt.errorbar(plot_x, plot_y, xerr=0., yerr=error_y, linewidth=1)
            plt.ylabel('AUC')
            plt.xlabel('click data')
            plt.savefig(app.config['STATISTICS_FOLDER'] + "/AUC_vs_data.png", bbox_inches='tight')

            plt.clf()
            #y_av = moving_average(plot_y2, 5)
            #plt.plot(plot_x, y_av, "r")

            plt.title('Precision as a function of click data objects')
            plt.errorbar(plot_x, plot_y2, xerr=0., yerr=error_y2, linewidth=1)
            plt.ylabel('Precision')
            plt.xlabel('click data')
            plt.savefig(app.config['STATISTICS_FOLDER'] + "/precision_vs_data.png", bbox_inches='tight')
    return 


def get_tf_idf_ranking(papers, user):
    ''' This function is used to rank papers by user interest when the user selects a date on daily papers '''
    X1, X2, X3 = prepare_data(papers, user, read=True)
    X_pred = np.c_[X1, X2, X3]

    model_file = app.config['ML_FOLDER'] + "/recommendation_model_%d.pickle" % user.id
    forest = pickle.load( open( model_file, "rb" ) )

    # Use the model for prediction
    y_proba = forest.predict_proba(X_pred)

    return y_proba[:,1]


def display_features(vectorizer, tfidf_result):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        print "{0:50} Score: {1}".format(item[0], item[1])
    return 


def moving_average(interval, window_size):
    if len(interval) > window_size:
        window = np.ones(int(window_size))/float(window_size)
        return list(np.convolve(interval, window, 'same'))
    else:
        return interval


def paper_click_data(user):
    # First we retrieve the papers... we need the title, abstract and authors to train the model
    input_papers = []
    for paper_click in user.paper_click_objects:
        res = []
        if paper_click.database_name == 'arxiv':
            try:
                res = es.get(index='paper-index', doc_type='paper', id=paper_click.paper_id)
            except Exception, e:
                print "Error: paper_click_data -- when retrieving paper from arxiv database", paper_click.paper_id
                print "Error message: ", str(e)
        else:
            try:
                res = es.get(index='pubmed-paper-index', doc_type='pubmed_paper', id=paper_click.paper_id)
            except Exception, e:
                print "Error: paper_click_data -- when retrieving paper from pubmed database", paper_click.paper_id
                print "Error message: ", str(e)
        if res:
            paper = models.Papers()
            # We only need the title, abstract and author list, since we do not use any other features (yet)
            paper.arxiv_title = res['_source']['title']
            paper.arxiv_abstract = res['_source']['abstract']
            paper.arxiv_authors = ', '.join(res['_source']['authors'])
            input_papers.append(paper)
        else:
            print "ERROR: paper not found?", paper_click.paper_id
    return input_papers


def get_negative_arxiv_papers(user, number_of_papers):
    # Prepare some filters
    or_filters = []
    and_filters = []
    for category in user.query_selection:
        or_filters.append(models.Papers.primary_category_id == category.category_id)
    # Make sure that the negative papers are not contained in the positive ones
    for paper_click in user.paper_click_objects:
        and_filters.append(models.Papers.arxiv_pure_id != paper_click.paper_id)
    # Here we pick the negative examples, meaning papers which the user did not select
    return models.Papers.query.filter(models.Papers.arxiv_published_datetime > user.start_datetime)\
                              .filter(or_(*or_filters))\
                              .filter(and_(*and_filters))\
                              .order_by(func.random())\
                              .limit(number_of_papers).all()


def get_auc_through_cross_validation(user):
    ''' 
    This function calculates the area under the curve (AUC) for the paper prediction model
    '''
    input_papers = paper_click_data(user)
    negative_papers = get_negative_arxiv_papers(user, len(input_papers))

    paper_sample = input_papers + negative_papers
    y = [1]*len(input_papers) + [0]*len(negative_papers)
    y = np.array(y)
    
    X1, X2, X3 = prepare_data(paper_sample, user, write=False)
    X = np.c_[X1, X2, X3]
    
    metrics = { 
        'num_cases': len(X),   
        'curves': [],
        'aucs': [],
        'pr': []
    }
    cross_validation_steps = 10
    kf = KFold(n_splits=cross_validation_steps, shuffle=True)
    forest = RandomForestClassifier(n_estimators=100)

    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        forest.fit(X_train, y_train)
        probabilities = forest.predict_proba(X_test)[:,1]
        precision, recall, thresholds = precision_recall_curve(y_test, probabilities, pos_label=1)

        thresholds = np.append(thresholds, np.array([1]))

        false_positive_rate, true_positive_rate, thresholds2 = roc_curve(y_test, probabilities, pos_label=1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print roc_auc
        if not math.isnan(roc_auc):
            av_pr = average_precision_score(y_test, probabilities)

            case_rate = []
            for threshold in thresholds:
                case_rate.append(np.mean(probabilities >= threshold))

            curves = {
                'thresholds': thresholds,
                'precision': precision,
                'recall': recall,
                'case_rate': case_rate,
                'fpr': false_positive_rate,
                'tpr': true_positive_rate
            }
            metrics['curves'].append(curves)
            metrics['aucs'].append(roc_auc)
            metrics['pr'].append(av_pr)

    plot_cross_validation_result(user, metrics)
    return metrics


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # remove numbers
    text_without_numbers = re.sub(r'\d+', '', text)
    # remove single letters
    text_without_single_letters = re.sub(r'\b\w\b', ' ', text_without_numbers)
    # remove punctuations 
    text_without_punctuation = ' '.join(re.findall('\w+', text_without_single_letters))
    tokens = nltk.word_tokenize(text_without_punctuation)
    stems = stem_tokens(tokens, stemmer)
    return stems


def pre_precess_text(list_of_texts):
    output_list_of_texts = []
    for text in list_of_texts:
        output_list_of_texts.append(' '.join( tokenize(text) ) )
    return output_list_of_texts


def prepare_data(papers, user, write=False, read=False):
    ''' 
    This function turns a list of papers into term frequency inverse document frequency 
    (tfidf) features.
    '''
    # Now we extract the features from these papers (abstract, title and authors)
    list_of_abstracts = [paper.arxiv_abstract for paper in papers]
    list_of_titles = [paper.arxiv_title for paper in papers]
    list_of_authors = [paper.arxiv_authors for paper in papers]

    # We pre-process the abstracts and titles, including stemming
    list_of_abstracts = pre_precess_text(list_of_abstracts)
    list_of_titles = pre_precess_text(list_of_titles)
    
    abstract_features_file = app.config['ML_FOLDER'] + "/abstract_features_%d.pickle" % user.id
    title_features_file = app.config['ML_FOLDER'] + "/title_features_%d.pickle" % user.id
    author_features_file = app.config['ML_FOLDER'] + "/author_features_%d.pickle" % user.id
    
    # Here we read in the feature lists or create new once if read = False
    if read:
        vectorizer = pickle.load( open( abstract_features_file, "rb" ) )
        train_data_features1 = vectorizer.transform(list_of_abstracts)
        vectorizer = pickle.load( open( title_features_file, "rb" ) )
        train_data_features2 = vectorizer.transform(list_of_titles)
        vectorizer = pickle.load( open( author_features_file, "rb" ) )
        train_data_features3 = vectorizer.transform(list_of_authors)
    else:
        # We calculate the tf-idf for each feature
        vectorizer = TfidfVectorizer(analyzer="word", tokenizer=None, stop_words="english", max_features=5000) 
        # fit_transform() serves two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of strings.
        train_data_features1 = vectorizer.fit_transform(list_of_abstracts)
        if write:
            pickle.dump(vectorizer, open( abstract_features_file, "wb" ), protocol=2)
        train_data_features2 = vectorizer.fit_transform(list_of_titles)
        if write:
            pickle.dump(vectorizer, open( title_features_file, "wb" ), protocol=2)
        train_data_features3 = vectorizer.fit_transform(list_of_authors)
        if write:
            pickle.dump(vectorizer, open( author_features_file, "wb" ), protocol=2)

    # Numpy arrays are easy to work with, so convert the results to arrays
    train_data_features1 = train_data_features1.toarray()
    train_data_features2 = train_data_features2.toarray()
    train_data_features3 = train_data_features3.toarray()

    return train_data_features1, train_data_features2, train_data_features3


def calc_model(user):
    ''' Calculate a new model (needs to be called within app.app_context())'''
    input_papers = paper_click_data(user)
    list_of_paper_click_paper_ids = [paper_click.paper_id for paper_click in user.paper_click_objects]
    if input_papers:
        # Now we get the negative paper examples
        if user.user_preference_database == 0:
            negative_papers = get_negative_arxiv_papers(user, len(input_papers))
        else:
            print 'pubmed user', user.id
            negative_papers = []

        paper_sample = input_papers + negative_papers
        y = [1]*len(input_papers) + [0]*len(negative_papers)
        y = np.array(y)
        
        X1, X2, X3 = prepare_data(paper_sample, user, write=True)
        X = np.c_[X1, X2, X3]
        X_train = X[:len(paper_sample)]
        X_pred = X[len(paper_sample):]

        forest = RandomForestClassifier(n_estimators=100)
        # forrest now stores the fit information used later for the prediction
        forest.fit(X_train, y)

        model_file = app.config['ML_FOLDER'] + "/recommendation_model_%d.pickle" % user.id
        pickle.dump(forest, open( model_file, "wb" ), protocol=2)
        return forest
    else:
        print "No input papers found... we only get here if there was aproblem with the paper retrieval...", user.id
        return False


def write_recommendations(user, papers, database_name):
    # remove all old recommendations
    while user.paper_recommendations.count():
        db.session.delete(user.paper_recommendations.first())
    # write new recommendations
    for paper in papers:
        recom = models.Recommendation(user_id = user.id,
                                      paper_id = paper[0].arxiv_pure_id,
                                      database_name = database_name,
                                      score = paper[1])
        db.session.add(recom)
    return 


def get_most_popular_papers(users):
    ''' 
    This function finds the most popular papers and produces paper recommendations 
    for users with too little data to train a machine learning model.
    '''
    # common timeframe for all recommendations
    timerange = datetime.datetime.utcnow() - datetime.timedelta(days=app.config['RECOMMENDATIONS_TIMEFRAME'])
    # Select the most popular papers for arXiv and pubmed by counting the paper click element
    arxiv_sub_query1 = db.session.query(models.Paperclick.paper_id, func.count(models.Paperclick.paper_id).label('click_counts'))\
                                       .filter(models.Paperclick.database_name=='arxiv')\
                                       .filter(models.Paperclick.click_datetime > timerange)\
                                       .group_by(models.Paperclick.paper_id).subquery()
    pubmed_sub_query1 = db.session.query(models.Paperclick.paper_id, func.count(models.Paperclick.paper_id).label('click_counts'))\
                                        .filter(models.Paperclick.database_name=='pubmed')\
                                        .filter(models.Paperclick.click_datetime > timerange)\
                                        .group_by(models.Paperclick.paper_id).subquery()
    for user in users: 
        print "process user %d" % user.id
        # For arXiv users
        if user.user_preference_database == 0:
            or_filters = []
            and_filters = []
            # If the user has selected arXiv categories, limit the recommendations to those categories
            for category in user.query_selection:
                or_filters.append(models.Papers.primary_category_id == category.category_id)
            # Exclude papers the user has already seen
            for paper_click in user.paper_click_objects:
                and_filters.append(models.Papers.arxiv_pure_id != paper_click.paper_id)
            # Join paper click objects with the papers table but limit the result to the 30 most popular papers
            # The Paperclick.paper_id is not unique (=arxiv_pure_id) and therefore we need the distinct clause
            papers = models.Papers.query.join(arxiv_sub_query1, arxiv_sub_query1.c.paper_id == models.Papers.arxiv_pure_id)\
                                             .filter(or_(*or_filters))\
                                             .filter(and_(*and_filters))\
                                             .order_by(arxiv_sub_query1.c.click_counts.desc())\
                                             .distinct(arxiv_sub_query1.c.click_counts, models.Papers.arxiv_pure_id)\
                                             .limit(app.config['NUM_RECOMMENDATIONS'])
            papers = papers.add_column(arxiv_sub_query1.c.click_counts).all()
            write_recommendations(user, papers, 'arxiv')
            db.session.commit()
        # For pubmed users
        elif user.user_preference_database == 1:
            # not implemented yet
            pass
    return


def get_recommendations(users, papers, pubmed_papers=[]):
    ''' 
    This function uses the available Paperclick data to train a machine learning model and 
    produce a list of paper recommendations for each user.
    '''
    for user in users:
        print "process user %d" % user.id
        time0 = time.time()
        # Here we do not train the model, instead we load the model. The model is calculated in
        # daily.py and should exists for all users with Paperclick elements
        model_file = app.config['ML_FOLDER'] + "/model1_%d.pickle" % user.id
        if os.path.exists(model_file):
            try:
                list_of_category_strings = [category.category_full_string for category in user.query_selection]
                list_of_paper_ids = [paper_click.paper_id for paper_click in user.paper_click_objects]
                # remove papers which have already been selected by the user
                if user.user_preference_database == 0:
                    my_set = [paper for paper in papers if paper.arxiv_pure_id not in list_of_paper_ids and paper.arxiv_primary_category in list_of_category_strings]
                    database_name = 'arxiv'
                else:
                    my_set = [paper for paper in pubmed_papers if paper.pm_id not in list_of_paper_ids]
                    database_name = 'pubmed'
                prob = get_tf_idf_ranking(my_set, user)
                paper_prob_pairs = [(x,y) for (y,x) in sorted(zip(prob, my_set), reverse=True)]
                write_recommendations(user, paper_prob_pairs[:app.config['NUM_RECOMMENDATIONS']], 'arxiv')
                db.session.commit()
                print "prediction finished after %f seconds" % (time.time() - time0)
            except Exception, e:
                print "Error: get_recommendations -- in user %d" % user.id
                print "error message: ", str(e)
    return 


def plot_cross_validation_result(user, results):
    # ROC Curve plot
    # average values for the legend
    auc_av = "{0:.2f}".format(np.mean(results['aucs']))
    auc_sd = "{0:.2f}".format(np.std(results['aucs']))

    plt.clf()
    plt.figure(2)
    # plot each individual ROC curve
    for chart in results['curves']:
        plt.plot(chart['fpr'], chart['tpr'], color='b', alpha=0.5)
    plt.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Cross-validated ROC for user %d (sample size %d)' % (user.id, len(user.paper_click_objects)))
    plt.text(0.6, 0.1,r'AUC = {av} \pm {sd}'.format(av=auc_av,sd=auc_sd))
    plt.savefig(app.config['STATISTICS_FOLDER'] + "/user_like_roc_curve2_%d.png" % user.id)

    # Precision-recall plot
    pr_av = "{0:.2f}".format(np.mean(results['pr']))
    pr_sd = "{0:.2f}".format(np.std(results['pr']))

    plt.clf()
    plt.figure(4)
    for chart in results['curves']:
        plt.plot(chart['recall'], chart['precision'], color='b', alpha=0.5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Cross-validated precision/recall for user %d (sample size %d)' % (user.id, len(user.paper_click_objects)))
    plt.text(0.6, 0.9,r'AUC = {av} \pm {sd}'.format(av=pr_av,sd=pr_sd))
    plt.savefig(app.config['STATISTICS_FOLDER'] + "/user_like_precision_recall_%d.png" % user.id)
    return 
