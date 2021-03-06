
/*==========================================================================
 * Copyright (c) 2004 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

//
// RandomWalkModel
//
// 23 June 2005 -- tds
//

#include "indri/RandomWalkModel.hpp"
#include <math.h>

//
// RandomWalkModel
//

indri::query::RandomWalkModel::RandomWalkModel(
                  indri::api::QueryEnvironment& environment,
                  const std::string& smoothing,
                  int maxGrams,
                  int documents )
  :
  _environment(environment),
  _smoothing(smoothing),
  _documents(documents),
  _maxGrams(maxGrams)
{
}

//
// ~RandomWalkModel
//

indri::query::RandomWalkModel::~RandomWalkModel() {
  HGram::iterator iter;

  for( iter = _gramTable.begin(); iter != _gramTable.end(); iter++ ) {
    delete *(iter->second);
  }
}

//
// getGrams
//

const std::vector<indri::query::RandomWalkModel::Gram*>& indri::query::RandomWalkModel::getGrams() const {
  return _grams;
}

//
// getQueryResults
//

const std::vector<indri::api::ScoredExtentResult>& indri::query::RandomWalkModel::getQueryResults() const {
  return _results;
}

//
// _extractDocuments
//

void indri::query::RandomWalkModel::_extractDocuments() {
  for( size_t i=0; i<_results.size(); i++ ) {
    _documentIDs.push_back( _results[i].document );
  }
}

//
// _countGrams
//
// Builds a hash table of grams, and counts the times that each
// gram occurs in each query result.
//

bool isValidWord(const string & word)
{
  size_t length = word.size();
  const char * chArray = word.c_str();
  size_t pos = 0;

  while (pos < length)
  {
    if(isalnum((unsigned char)*(chArray+pos)) == 0)
    {
      return false;
    }
    pos ++;
  }
  return true;
}

void indri::query::RandomWalkModel::_countGrams() {
  // for each query result
  for( size_t i=0; i<_results.size(); i++ ) {
    // run through the text, extracting n-grams
    indri::api::ScoredExtentResult& result = _results[i];
    indri::api::DocumentVector* v = _vectors[i];
    std::vector<int>& positions = v->positions();
    std::vector<std::string>& stems = v->stems();
    if (result.end == 0) result.end = positions.size();

    // for each word position in the text
    for( int j = result.begin; j < result.end; j++ ) {
      int maxGram = std::min( _maxGrams, result.end - j );

      // extract every possible n-gram that starts at this position
      // up to _maxGrams in length
      for( int n = 1; n <= maxGram; n++ ) {
        GramCounts* newCounts = new GramCounts;
        bool containsOOV = false;

        // build the gram
        for( int k = 0; k < n; k++ ) {
          if( positions[ k + j ] == 0 || (! isValidWord(stems[ positions[ k + j ] ])) ) {
            containsOOV = true;
            break;
          }

          newCounts->gram.terms.push_back( stems[ positions[ k + j ] ] );
        }

        if( containsOOV ) {
          // if this contanied OOV, all larger n-grams
          // starting at this point also will
          delete newCounts;
          break;
        }



				
        GramCounts** gramCounts = 0;
        gramCounts = _gramTable.find( &newCounts->gram );
        size_t* curcount = 0;
	curcount = _gramCounts.find(&newCounts->gram);
        if( gramCounts == 0 ) {
          _gramTable.insert( &newCounts->gram, newCounts );
          _gramCounts.insert(&newCounts->gram,1);
          gramCounts = &newCounts;
        } else {
                if (curcount == 0)
			fprintf (stderr, "Errrrrror \n");
        	_gramCounts.insert(&newCounts->gram,1 +  (*curcount));
          delete newCounts;
        }

        if( (*gramCounts)->counts.size() && (*gramCounts)->counts.back().first == i ) {
          // we already have some counts going for this query result, so just add this one
          (*gramCounts)->counts.back().second++;
        } else {
          // no counts yet in this document, so add an entry
          (*gramCounts)->counts.push_back( std::make_pair( i, 1 ) );
        }
      }
    }
  }
}


//
// _scoreGrams
//

void indri::query::RandomWalkModel::_buildCoocMatrix() {
   
  // count the number of grams which occur in the dataset at least > limit_count types
  int errcount = 0;
  size_t limit_count = 0;
  HGramCount::iterator iter_count;
  valid_grams_count = 0;
  total_score_count = 0;
  cout << "gram counts length " << _gramCounts.size() << endl;
  for (iter_count = _gramCounts.begin(); iter_count != _gramCounts.end() ; iter_count++)
        { // cout << "doc_grams" <<  (*iter_count->first)->term_string() << " " <<  *iter_count->second << endl;
	  if (*iter_count->second > limit_count) {
  //		fprintf(stderr,"l2\n");
		 _idGrams.insert(valid_grams_count,*iter_count->first);
				
		// fprintf(stderr,"l3\n");
		 _gramIds.insert(*iter_count->first,valid_grams_count);
		 valid_grams_count++;
		 std::string gram_text = (*iter_count->first)->term_string();
		 int m = 0;
		 for (int i = 0 ; i < _queryGrams.size() ; i++) {

		 		if (_queryGrams[i].compare(gram_text) == 0) {
		 			    cout << "matched gram " << gram_text << endl;
		 		        total_score_count += 1.0; m = 1;
		 			    _gramScores.insert((*iter_count->first),1.0);
		 		}

		  	 }
		 if (m == 0) _gramScores.insert((*iter_count->first),0);
	  }

  }
  fprintf(stderr,"h1\n");

  // Initialize co-occurrence matrix
  _cooccurMatrix = (size_t**) malloc(valid_grams_count * sizeof(size_t*));
  for (int i = 0; i < valid_grams_count; i++) {
	  _cooccurMatrix[i] = (size_t*) malloc(valid_grams_count*sizeof(size_t));
      for (int j = 0; j < valid_grams_count; j++)
    	  _cooccurMatrix[i][j] = 0;
  }

  fprintf(stderr,"h1\n");
  HGram::iterator iter;

  for( iter = _gramTable.begin(); iter != _gramTable.end(); iter++ ) {
	  Gram* gram = *iter->first;
	  GramCounts* gramCounts = *iter->second;
	  size_t* secondGramId = 0;
	  secondGramId = _gramIds.find(gram);
	  if (secondGramId == 0) continue;
	  for( size_t i=0; i< gramCounts->counts.size(); i++ ) {
		  int docid = gramCounts->counts[i].first;
		  int count = gramCounts->counts[i].second;
		  VectorGramCounts* vcg = 0;
		  vcg = _docIdGrams.find(docid);
		  if (vcg == 0) {
			  VectorGramCounts new_vcg;
			  new_vcg.push_back(std::make_pair(gram,count));
			  _docIdGrams.insert(docid, new_vcg);
		  }
		  else {
			  vcg->push_back(std::make_pair(gram,count));

		  }
	  }
  }

  fprintf(stderr,"h1\n");
  for (size_t i = 0; i < valid_grams_count; i++) {
	 Gram** g1 = _idGrams.find(i);
	 GramCounts** g1c = _gramTable.find(*g1);
	 for (size_t k = 0; k < (*g1c)->counts.size() ; k++) {
		 int docid = (*g1c)->counts[k].first;
		 VectorGramCounts* vcg = _docIdGrams.find(docid);
		 for (size_t j = 0 ; j < vcg->size(); j++)
			 _cooccurMatrix[i][j] +=(*g1c)->counts[k].second * (*vcg)[j].second;
	 }

  }



  fprintf(stderr,"hn\n");

}


//
// _scoreGrams
//

void indri::query::RandomWalkModel::_scoreGrams() {
  HGram::iterator iter;
  double collectionCount = (double)_environment.termCount();
  indri::query::TermScoreFunction* function = 0;

  // for each gram we've seen
  for( iter = _gramTable.begin(); iter != _gramTable.end(); iter++ ) {
    // gather the number of times this gram occurs in the collection
    double gramCount = 0;

    Gram* gram = *iter->first;
    GramCounts* gramCounts = *iter->second;

    if( _smoothing.length() != 0 ) {
      // it's only important to get background frequencies if
      // we're smoothing with them; otherwise we don't care.

      if( gram->terms.size() == 1 ) {
        gramCount = (double)_environment.stemCount( gram->terms[0] );
      } else {
        // notice that we're running a query here;
        // this is likely to be slow. (be warned)

        std::stringstream s;
        s << "#1( ";

        for( size_t i=0; i< gram->terms.size(); i++ ) {
          s << " \"" << gram->terms[i] << "\"" << std::endl;
        }

        s << ") ";
        gramCount = _environment.expressionCount( s.str() );
      }

      double gramFrequency = gramCount / collectionCount;
      //      function = indri::query::TermScoreFunctionFactory::get( _smoothing, gramFrequency );
      function = indri::query::TermScoreFunctionFactory::get( _smoothing, gramCount, collectionCount, 0 , 0 );
    }

    // now, aggregate scores for each retrieved item
    std::vector<indri::api::ScoredExtentResult>::iterator riter;
    double gramScore = 0;
    size_t c;
    size_t r;

    for( r = 0, c = 0; r < _results.size() && c < gramCounts->counts.size(); r++ ) {
      int contextLength = _results[r].end - _results[r].begin;
      // has been converted to a posterior probability.
      double documentScore = _results[r].score;
      double termScore = 0;
      double occurrences = 0;

      if( gramCounts->counts[c].first == r ) {
        // we have counts for this result
        occurrences = gramCounts->counts[c].second;
        c++;
      }

      // determine the score for this term
      if( function != 0 ) {
        // log probability here
        termScore = exp(function->scoreOccurrence( occurrences, contextLength ));
      } else {
        termScore = occurrences / double(contextLength);
      }
      //RMExpander weights this by 1/fbDocs
      // Unclear as to why.
      gramScore += documentScore * termScore;
      //gramScore += (1.0/_documents) * documentScore * termScore;
    }

    gram->weight = gramScore;
    delete function;
  }
}

//
// _sortGrams
//

void indri::query::RandomWalkModel::_sortGrams() {
  // copy grams into a _grams vector
  HGram::iterator iter;
  _grams.clear();

  for( iter = _gramTable.begin(); iter != _gramTable.end(); iter++ ) {
    _grams.push_back( *(iter->first) );
  }

  std::sort( _grams.begin(), _grams.end(), Gram::weight_greater() );
}

// In:  log(x1) log(x2) ... log(xN)
// Out: x1/sum, x2/sum, ... xN/sum
//
// Extra care is taken to make sure we don't overflow
// machine precision when taking exp (log x)
// This is done by subtracting a constant K which cancels out
// Right now K is set to maximally preserve the highest value
// but could be altered to a min or average, or whatever...

static void _logtoposterior(std::vector<indri::api::ScoredExtentResult> &res) {
  if (res.size() == 0) return;
  std::vector<indri::api::ScoredExtentResult>::iterator iter;
  iter = res.begin();
  double K = (*iter).score;
  // first is max
  double sum=0;

  for (iter = res.begin(); iter != res.end(); iter++) {
    sum += (*iter).score=exp((*iter).score - K);
  }
  for (iter = res.begin(); iter != res.end(); iter++) {
    (*iter).score/=sum;
  }
}


//
// generate
//

void indri::query::RandomWalkModel::generate( const std::string& query ) {
  try {
    // run the query, get the document vectors
    _results = _environment.runQuery( query, _documents );
    _logtoposterior(_results);
    _grams.clear();
    _extractDocuments();
    _vectors = _environment.documentVectors( _documentIDs );

    _countGrams();
    fprintf(stderr, "here Amit \n");
    _buildCoocMatrix();
   // _scoreGrams();
    _sortGrams();
    for (unsigned int i = 0; i < _vectors.size(); i++)
      delete _vectors[i];
  } catch( lemur::api::Exception& e ) {
    LEMUR_RETHROW( e, "Couldn't generate relevance model for '" + query + "' because: " );
  }
}

//
// generate
//

vector<std::string> indri::query::RandomWalkModel::find_query_grams(std::string ans) {
vector<std::string> v  = tokenize_string(ans);
vector<std::string> ans_vector ;

for ( int i = 0 ; i < v.size(); i++) {
	for (int j = 1 ; j <= _maxGrams; j++) {
		std::string st = "";
		for (int k = 0 ; k < j ; k++)
			if (i+k < v.size())
				st += " " + v[i + k];
			if (st.length() > 1)
				{
				ans_vector.push_back(st.substr(1));
				cout << "query grams " << st.substr(1) << endl;
				}



	}


}
 fprintf (stderr, "got vector \n");
 return ans_vector;
}

vector<std::string> indri::query::RandomWalkModel::tokenize_string(std::string ans) {
vector<std::string> v ;

while (ans.length() > 0) {
  int l = ans.find (" ");
  if ( l == -1) {
   v.push_back(ans);
   break;
  }
  else if (l != 0) {
   std:string st = ans.substr(0,l);
   if (st.length() > 0 )
        v.push_back(st);
   ans = ans.substr(l);
    }
    else {

   ans = ans.substr(1);
    }

}
return v;
}

void indri::query::RandomWalkModel::_computePageRank() {
	HGramScore::iterator iter_score;
          cout << "Sizeee " << _gramScores.size() << endl;
	  int count = 0;  
	for (iter_score = _gramScores.begin(); iter_score != _gramScores.end() ; iter_score++)
		  	  {
	count++;
      cout << "Sizeee " << _gramScores.size() << endl;
        _gramScores.insert(*iter_score->first , *iter_score->second / total_score_count);
	cout << count << endl;
	}
	  int number_of_iterations = 500;
	  double lambda = 0.3;
          cout << "Sizeee2 " << _gramScores.size() << endl;
	  for (int i = 0 ; i < number_of_iterations; i++) {
		   fprintf (stderr, "number of iterations pagerank completed %d\n",i);
		  for (iter_score = _gramScores.begin(); iter_score != _gramScores.end() ; iter_score++) {
			  double new_score = 0;
			  double cur_score = *iter_score->second;
			  Gram * gm = *iter_score->first;
			  size_t *gid = _gramIds.find(gm);
			  for (size_t j = 0 ; j < valid_grams_count; j++) {
				  Gram** gm2 = _idGrams.find(j);
				  size_t wt = _cooccurMatrix[*gid][j];
				  new_score += wt * (*_gramScores.find(*gm2));
			  }
			  new_score = lambda * cur_score + (1 - lambda ) * new_score;
			  _gramScores.insert(gm, new_score);
		  }
	  }

          cout << "Sizeee3 " << _gramScores.size() << endl;

	  for (iter_score = _gramScores.begin(); iter_score != _gramScores.end() ; iter_score++)
	  		  	{ cout << "pr scores " << (*iter_score->first)->term_string() << " " << (*iter_score->first)->weight << endl;  
				(*iter_score->first)->weight = *iter_score->second ;
				
				}

}


void indri::query::RandomWalkModel::generate( const std::string& query, const std::vector<indri::api::ScoredExtentResult>& results  ) {
  try {
    fprintf (stderr, "hey1 \n");
    _results = results;
    _logtoposterior(_results);
    _grams.clear();
    _extractDocuments();
    _vectors = _environment.documentVectors( _documentIDs );
    _queryGrams = find_query_grams(query);
    fprintf(stderr, "here Amit2 \n");
    _countGrams();

    fprintf(stderr, "here Amit2 \n");
   _buildCoocMatrix();
   _computePageRank();
   // _scoreGrams();
    fprintf(stderr, "here Amit3 \n");
    _sortGrams();
    fprintf(stderr, "here Amit4 \n");
    for (unsigned int i = 0; i < _vectors.size(); i++)
      delete _vectors[i];
    fprintf(stderr, "here Amit4 \n");
  } catch( lemur::api::Exception& e ) {
    LEMUR_RETHROW( e, "Couldn't generate relevance model for '" + query + "' because: " );
  }
}

