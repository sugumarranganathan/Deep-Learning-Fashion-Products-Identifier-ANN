[07:50:58] 🐍 Python dependencies were installed from /mount/src/deep-learning-fashion-products-identifier-ann/requirements.txt using uv.

Check if streamlit is installed

Streamlit is already installed

[07:50:59] 📦 Processed dependencies!




2026-05-27 07:51:38.410938: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.

To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:600 in _run_script                                      

                                                                                

  /mount/src/deep-learning-fashion-products-identifier-ann/app.py:77 in         

  <module>                                                                      

                                                                                

     74 # -----------------------------------                                   

     75 # Initialize Model                                                      

     76 # -----------------------------------                                   

  ❱  77 model = load_model_cached()                                             

     78                                                                         

     79 # -----------------------------------                                   

     80 # Fashion Product Labels                                                

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/caching/  

  cache_utils.py:165 in wrapper                                                 

                                                                                

    162 │                                                                       

    163 │   @functools.wraps(info.func)                                         

    164 │   def wrapper(*args, **kwargs):                                       

  ❱ 165 │   │   return cached_func(*args, **kwargs)                             

    166 │                                                                       

    167 │   # Give our wrapper its `clear` function.                            

    168 │   # (This results in a spurious mypy error that we suppress.)         

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/caching/  

  cache_utils.py:194 in __call__                                                

                                                                                

    191 │   │                                                                   

    192 │   │   if self._info.show_spinner or isinstance(self._info.show_spinn  

    193 │   │   │   with spinner(message, _cache=True):                         

  ❱ 194 │   │   │   │   return self._get_or_create_cached_value(args, kwargs)   

    195 │   │   else:                                                           

    196 │   │   │   return self._get_or_create_cached_value(args, kwargs)       

    197                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/caching/  

  cache_utils.py:221 in _get_or_create_cached_value                             

                                                                                

    218 │   │   │   return self._handle_cache_hit(cached_result)                

    219 │   │   except CacheKeyNotFoundError:                                   

    220 │   │   │   pass                                                        

  ❱ 221 │   │   return self._handle_cache_miss(cache, value_key, func_args, fu  

    222 │                                                                       

    223 │   def _handle_cache_hit(self, result: CachedResult) -> Any:           

    224 │   │   """Handle a cache hit: replay the result's cached messages, an  

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/streamlit/runtime/caching/  

  cache_utils.py:277 in _handle_cache_miss                                      

                                                                                

    274 │   │   │   with self._info.cached_message_replay_ctx.calling_cached_f  

    275 │   │   │   │   self._info.func, self._info.allow_widgets               

    276 │   │   │   ):                                                          

  ❱ 277 │   │   │   │   computed_value = self._info.func(*func_args, **func_kw  

    278 │   │   │                                                               

    279 │   │   │   # We've computed our value, and now we need to write it ba  

    280 │   │   │   # along with any "replay messages" that were generated dur  

                                                                                

  /mount/src/deep-learning-fashion-products-identifier-ann/app.py:68 in         

  load_model_cached                                                             

                                                                                

     65 │   model.build((None, 784))                                            

     66 │                                                                       

     67 │   # Load trained weights                                              

  ❱  68 │   model.load_weights(                                                 

     69 │   │   "fashion_model_fixed.h5"                                        

     70 │   )                                                                   

     71                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/keras/utils/traceback_util  

  s.py:70 in error_handler                                                      

                                                                                

     67 │   │   │   filtered_tb = _process_traceback_frames(e.__traceback__)    

     68 │   │   │   # To get the full stack trace, call:                        

     69 │   │   │   # `tf.debugging.disable_traceback_filtering()`              

  ❱  70 │   │   │   raise e.with_traceback(filtered_tb) from None               

     71 │   │   finally:                                                        

     72 │   │   │   del filtered_tb                                             

     73                                                                         

                                                                                

  /home/adminuser/venv/lib/python3.10/site-packages/keras/saving/legacy/hdf5_f  

  ormat.py:826 in load_weights_from_hdf5_group                                  

                                                                                

     823 │   │   │   layer, weight_values, original_keras_version, original_ba  

     824 │   │   )                                                              

     825 │   │   if len(weight_values) != len(symbolic_weights):                

  ❱  826 │   │   │   raise ValueError(                                          

     827 │   │   │   │   f"Weight count mismatch for layer #{k} (named {layer.  

     828 │   │   │   │   f"the current model, {name} in the save file). "       

     829 │   │   │   │   f"Layer expects {len(symbolic_weights)} weight(s). Re  

────────────────────────────────────────────────────────────────────────────────

ValueError: Weight count mismatch for layer #1 (named dense_1 in the current 

model, batch_normalization in the save file). Layer expects 2 weight(s). 

Received 4 saved weight(s)
