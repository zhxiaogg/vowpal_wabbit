#include<unordered_map>
#include "reductions.h"

namespace FAIR
{
  using namespace std;
  struct fair
  {
    unsigned char fair_space;
    uint32_t k;
    
    unordered_map<uint64_t, uint64_t> attribute_counts;
    unordered_map<uint64_t, float> lambdas;
    double average_lambda;
    uint64_t event_count;

    COST_SENSITIVE::label cs_label;
  };


  void record_lambda(fair& data, example& ec)
  { //record lambda values
    if (ec.feature_space[data.fair_space].size() == 0)
      THROW("A lambda example must have 1 or more protected attributes.");
    
    data.average_lambda = 0.;
    for (features::iterator& f: ec.feature_space[data.fair_space])
      {
	uint64_t key = f.index() + ec.ft_offset;
	data.lambdas.emplace(key,f.value());
	data.lambdas[key] = f.value();
	data.average_lambda = f.value();
      }
    data.average_lambda /= ec.feature_space[data.fair_space].size();
  }
  
  template <bool is_learn>
  void predict_or_learn(fair& data, LEARNER::base_learner& base, example& ec)
  {
    if (ec.tag.size() == 6 && ec.tag[0] == 'l'  && ec.tag[1] == 'a' && ec.tag[2] == 'm' && ec.tag[3] == 'b' && ec.tag[4] == 'd' && ec.tag[5] == 'a')
      {
	record_lambda(data,ec);
	return;
      }
	
    data.cs_label.costs.erase();
    MULTICLASS::label_t mc_label = ec.l.multi;
    
    if (!is_learn)
      {
	ec.l.cs = data.cs_label;
	base.predict(ec);
      }
    else
      {
	for (features::iterator& f : ec.feature_space[data.fair_space]) //update stats
	  {
	    uint64_t key = f.index() + ec.ft_offset;
	    data.attribute_counts.emplace(key,0);//insert 0 if it does not exist
	    data.lambdas.emplace(key,0);//insert 0 if it does not exist
	    data.attribute_counts[key]++;//count the key values.
	  }
	data.event_count++;
	
	//construct cost sensitive label
	for (uint32_t i = 1; i <= data.k; i++)
	  { COST_SENSITIVE::wclass wc = {0., i, 0., 0.};
	    if (i == mc_label.label)
	      wc.x = 0.;
	    else
	      wc.x = 1.;

	    if (i==2)
	      if (ec.feature_space[data.fair_space].size() > 0)
		{
		  uint64_t key = ec.feature_space[data.fair_space].indicies[0] + ec.ft_offset;
		  
		  float attribute_fraction = (float) data.attribute_counts[key] / (float) data.event_count; 
		  
		  wc.x += (data.lambdas[key] - data.average_lambda) / attribute_fraction;
		}
	    
	    data.cs_label.costs.push_back(wc);
	  }
	
	ec.l.cs = data.cs_label;
	base.learn(ec);
      }
    ec.l.multi = mc_label;
  }
}

using namespace FAIR;
  
LEARNER::base_learner* fair_setup(vw& all)
{ if (missing_option<uint32_t, true>(all, "fair", "make classification fair with respect to some attributes for <k> classes"))
    return nullptr;
  new_options(all)
    //    ("lambda", po::value< vector<string> >()->default_value(0.f), "lagrangian parameter") Instead of this, datasets must be prefaced with a special example that specifies the lambda value associated with each attribute
    ("space", po::value<char>(), "protected attribute space");

  add_options(all);

  fair& data = calloc_or_throw<fair>();

  data.k = (uint32_t)all.vm["fair"].as<uint32_t>();
  if (all.vm.count("space"))
    data.fair_space = (char)all.vm["space"].as<char>();

  if (count(all.args.begin(), all.args.end(),"--csoaa") == 0)
  { all.args.push_back("--csoaa");
    stringstream ss;
    ss << data.k;
    all.args.push_back(ss.str());
  }

  LEARNER::learner<fair>& ret =
    init_multiclass_learner(&data, setup_base(all), predict_or_learn<true>, predict_or_learn<false>,all.p,1,prediction_type::multiclass);

  return make_base(ret);
}
