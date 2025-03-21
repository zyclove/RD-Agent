# Changelog

## [0.4.0](https://github.com/microsoft/RD-Agent/compare/v0.3.0...v0.4.0) (2025-03-21)


### Features

* (Kaggle) add base template for competition: tabular-playground-series-may-2022 ([#481](https://github.com/microsoft/RD-Agent/issues/481)) ([f3405ca](https://github.com/microsoft/RD-Agent/commit/f3405ca732eb0ddca8e18ea72f69cbd86055c4ab))
* a unified CoSTEER to fit more scenarios ([#491](https://github.com/microsoft/RD-Agent/issues/491)) ([cddbd02](https://github.com/microsoft/RD-Agent/commit/cddbd02e3ad3ccf6ad01443777319dc5c7eb08a7))
* add a new competition ([#474](https://github.com/microsoft/RD-Agent/issues/474)) ([2fc0d77](https://github.com/microsoft/RD-Agent/commit/2fc0d77c485a31f647e21f4578e2e326f7032964))
* add baseline score stat ([#590](https://github.com/microsoft/RD-Agent/issues/590)) ([2948026](https://github.com/microsoft/RD-Agent/commit/2948026c390d067b643f8c8247c1447f1dc023e4))
* add configurable volume mode for Docker volumes in env.py ([#537](https://github.com/microsoft/RD-Agent/issues/537)) ([642a022](https://github.com/microsoft/RD-Agent/commit/642a02239431411b91959f23e69b454997ca75d5))
* add constraint labels for semantic search ([#680](https://github.com/microsoft/RD-Agent/issues/680)) ([0584cfc](https://github.com/microsoft/RD-Agent/commit/0584cfcd13ca1a62c85390ea2ee7574370748d31))
* add cross validation to workflow ([#700](https://github.com/microsoft/RD-Agent/issues/700)) ([82e9b00](https://github.com/microsoft/RD-Agent/commit/82e9b00be62b01673353a7aaa3ab0e2e3ecaf3ca))
* add do_truncate control for the load function ([#656](https://github.com/microsoft/RD-Agent/issues/656)) ([2b960a5](https://github.com/microsoft/RD-Agent/commit/2b960a58dfdeba69522a0f72ecf0975bb6ae87ee))
* add do_truncate control for the load function ([#656](https://github.com/microsoft/RD-Agent/issues/656)) ([2b960a5](https://github.com/microsoft/RD-Agent/commit/2b960a58dfdeba69522a0f72ecf0975bb6ae87ee))
* add eda to data science scenario ([#639](https://github.com/microsoft/RD-Agent/issues/639)) ([35aa479](https://github.com/microsoft/RD-Agent/commit/35aa479f00edf118d43ec228e0a84c155332957a))
* Add line length limit to shrink_text function and settings ([#715](https://github.com/microsoft/RD-Agent/issues/715)) ([75ed5e1](https://github.com/microsoft/RD-Agent/commit/75ed5e1c2ce1bf20bb55190c10a4134e04694d2b))
* add loop_n parameter to the main loop ([#611](https://github.com/microsoft/RD-Agent/issues/611)) ([778c166](https://github.com/microsoft/RD-Agent/commit/778c166962250e3b9e7ad85de37f62297d370b45))
* add max time config to costeer in data science ([#645](https://github.com/microsoft/RD-Agent/issues/645)) ([534686c](https://github.com/microsoft/RD-Agent/commit/534686c2ba7d9fa979c0762ad3177c36f6d7f4cb))
* add mlebench submission validitor ([#545](https://github.com/microsoft/RD-Agent/issues/545)) ([712d94a](https://github.com/microsoft/RD-Agent/commit/712d94a7d6f22187fc3d18bd434e71ec6997aa9f))
* add model removal and adjust some framework logic ([#681](https://github.com/microsoft/RD-Agent/issues/681)) ([1edf881](https://github.com/microsoft/RD-Agent/commit/1edf881c63512d351c0dd074d7a1c0965ff3119b))
* add output_path to load function of LoopBase ([#628](https://github.com/microsoft/RD-Agent/issues/628)) ([dd33726](https://github.com/microsoft/RD-Agent/commit/dd33726ac5de75dc2030d193d457d59490b3361e))
* add rank into report (mle_summary) ([#665](https://github.com/microsoft/RD-Agent/issues/665)) ([13f7922](https://github.com/microsoft/RD-Agent/commit/13f7922aaae9e4143aac4ad08ec1c556c2faf04e))
* add restart and fix unzip ([#538](https://github.com/microsoft/RD-Agent/issues/538)) ([ed2c7d1](https://github.com/microsoft/RD-Agent/commit/ed2c7d175f1f44ca06ad7a63b08da12f6c4df9ab))
* add retry mechanism with wait_retry decorator and refactor diff generation ([#572](https://github.com/microsoft/RD-Agent/issues/572)) ([de1cd72](https://github.com/microsoft/RD-Agent/commit/de1cd72f068ebd1e1bd5bc2ad2b12ae484d54831))
* add the shape of the CSV to the dataset description ([#561](https://github.com/microsoft/RD-Agent/issues/561)) ([a10c881](https://github.com/microsoft/RD-Agent/commit/a10c881bd86796e6167257ad26dd165f7e46d813))
* add timeout settings and cleanup step in data science runner ([#539](https://github.com/microsoft/RD-Agent/issues/539)) ([295abd5](https://github.com/microsoft/RD-Agent/commit/295abd56f7b58055bd27b247dfed47eb85e9b0cd))
* add type checker to api backend & align litellm and old backend ([#647](https://github.com/microsoft/RD-Agent/issues/647)) ([d38eae9](https://github.com/microsoft/RD-Agent/commit/d38eae986a0ba69d71288fa09fcc21e227551a02))
* align mlebench data and evaluation & several fix on kaggle workflow ([#477](https://github.com/microsoft/RD-Agent/issues/477)) ([f6c522b](https://github.com/microsoft/RD-Agent/commit/f6c522b651db3c1f6af6815347589917f46e433a))
* **backend:** integrate LiteLLM API Backend ([#564](https://github.com/microsoft/RD-Agent/issues/564)) ([f477687](https://github.com/microsoft/RD-Agent/commit/f4776879c76a213d53875b307c94be1ea5cfd9ba))
* base data science scenario UI ([#525](https://github.com/microsoft/RD-Agent/issues/525)) ([39917b3](https://github.com/microsoft/RD-Agent/commit/39917b354b22a8488a17396fe2245cb41e3def03))
* condaenv & full docker env ([#668](https://github.com/microsoft/RD-Agent/issues/668)) ([084dd6d](https://github.com/microsoft/RD-Agent/commit/084dd6d748a89492ea0888acb316b9bb9efeb62f))
* diff mode fix ([#569](https://github.com/microsoft/RD-Agent/issues/569)) ([0c509f5](https://github.com/microsoft/RD-Agent/commit/0c509f599ce19303b44d8192ec3eb634c24992d6))
* display LLM prompt ([#676](https://github.com/microsoft/RD-Agent/issues/676)) ([8c93bba](https://github.com/microsoft/RD-Agent/commit/8c93bba82e185edcf4204cc574df5f41bcdfa9d2))
* Dynamically find and use sample submission file in eval tests ([#542](https://github.com/microsoft/RD-Agent/issues/542)) ([5f12b44](https://github.com/microsoft/RD-Agent/commit/5f12b44c89dd26b250e914192f9beb2da38fb3ab))
* end-to-end optimization ([#473](https://github.com/microsoft/RD-Agent/issues/473)) ([d41343a](https://github.com/microsoft/RD-Agent/commit/d41343a63d87bf3479f5ec30745ea788580495bf))
* Enhance eval script with file cleanup and detailed submission checks ([#529](https://github.com/microsoft/RD-Agent/issues/529)) ([cf2ff92](https://github.com/microsoft/RD-Agent/commit/cf2ff9213d3a8b0fad64df7cae0c35f996d72e27))
* exclude invalid session log folder ([#554](https://github.com/microsoft/RD-Agent/issues/554)) ([fa86e4d](https://github.com/microsoft/RD-Agent/commit/fa86e4d1805000e0e5779c662ccbb5273fda623c))
* improve the framework's ability to adaptively adjust the model ([#629](https://github.com/microsoft/RD-Agent/issues/629)) ([93806f3](https://github.com/microsoft/RD-Agent/commit/93806f33a1e0f29a125e29303d4b984a9817c3c0))
* independent use_azure_token_provider on chat and embedding ([#452](https://github.com/microsoft/RD-Agent/issues/452)) ([d223004](https://github.com/microsoft/RD-Agent/commit/d223004917692e231b251330cbc8676081d5a10d))
* integrate azure deepseek r1 ([#591](https://github.com/microsoft/RD-Agent/issues/591)) ([e79ce5c](https://github.com/microsoft/RD-Agent/commit/e79ce5c38539138abe04eb9809fbde437e97bbb7))
* kaggle refactor ([#489](https://github.com/microsoft/RD-Agent/issues/489)) ([1b057d0](https://github.com/microsoft/RD-Agent/commit/1b057d0d63a861fba4b3cb59c6c5fc1a0e3da383))
* **kaggle:** several update in kaggle scenarios ([#476](https://github.com/microsoft/RD-Agent/issues/476)) ([245d211](https://github.com/microsoft/RD-Agent/commit/245d211dcbfb18ebcc554247a0e3a8dbecf6f3bd))
* Make system prompt role customizable in LLM settings ([#632](https://github.com/microsoft/RD-Agent/issues/632)) ([e4acd92](https://github.com/microsoft/RD-Agent/commit/e4acd92cc5eec6db5c29cb2d4788020fb89099b7))
* multi log folder, replace "epxx" in workspace path ([#555](https://github.com/microsoft/RD-Agent/issues/555)) ([8a69c9c](https://github.com/microsoft/RD-Agent/commit/8a69c9c9630860c9b644356e1f71654aea222328))
* new-york-city-taxi-fare-prediction_template ([#488](https://github.com/microsoft/RD-Agent/issues/488)) ([a9caab7](https://github.com/microsoft/RD-Agent/commit/a9caab7bc5dc86f395a008e523355922137aef17))
* out spec change for o1-preview ([#666](https://github.com/microsoft/RD-Agent/issues/666)) ([22894bd](https://github.com/microsoft/RD-Agent/commit/22894bdbee26b9cad73646d2975857787e515f75))
* refactor for general data science ([#498](https://github.com/microsoft/RD-Agent/issues/498)) ([7002dc4](https://github.com/microsoft/RD-Agent/commit/7002dc4981a4f72096b438d2fe4fd9ff268c54f3))
* refine logic for qlib_factor_from_report ([#463](https://github.com/microsoft/RD-Agent/issues/463)) ([21348d8](https://github.com/microsoft/RD-Agent/commit/21348d89e0e0eec1b4fab4e7a497f1eb34b8fe72))
* run benchmark on gpt-4o & llama 3.1 ([#497](https://github.com/microsoft/RD-Agent/issues/497)) ([64af0b5](https://github.com/microsoft/RD-Agent/commit/64af0b5529b687cce8b5b7a1893946e15edca626))
* summary and UI update ([#581](https://github.com/microsoft/RD-Agent/issues/581)) ([efa51f9](https://github.com/microsoft/RD-Agent/commit/efa51f9c259a06fe219f3137f0a1005e50d2bfdd))
* template changes for some kaggle competitions ([#484](https://github.com/microsoft/RD-Agent/issues/484)) ([2e38000](https://github.com/microsoft/RD-Agent/commit/2e38000091030811fc081d72016c7bbadf7efd50))
* variable printing tool of data_science coder testing ([#658](https://github.com/microsoft/RD-Agent/issues/658)) ([116c061](https://github.com/microsoft/RD-Agent/commit/116c06190b01f0b621c021726a1be23458ab1154))


### Bug Fixes

* a default conf in scen qlib ([#503](https://github.com/microsoft/RD-Agent/issues/503)) ([d64a228](https://github.com/microsoft/RD-Agent/commit/d64a228525cbedd7687c1e06132eacd0d0647697))
* a small bug in exp_gen ([#606](https://github.com/microsoft/RD-Agent/issues/606)) ([f734dde](https://github.com/microsoft/RD-Agent/commit/f734dde0b0101e13f38151468c8ddf9e23af26ac))
* add check when retrying gen model codes ([#699](https://github.com/microsoft/RD-Agent/issues/699)) ([3b82f15](https://github.com/microsoft/RD-Agent/commit/3b82f159474087902d3c6007d370e3282b549015))
* add DSExperiment type check and directory validation in log proc… ([#535](https://github.com/microsoft/RD-Agent/issues/535)) ([f59b12c](https://github.com/microsoft/RD-Agent/commit/f59b12c9cc9afde82b74bc133797ff1396678627))
* add ensemble test, change to "use cross-validation if possible" in workflow spec ([#634](https://github.com/microsoft/RD-Agent/issues/634)) ([acc97a8](https://github.com/microsoft/RD-Agent/commit/acc97a8217253497afedcfa829902b4432e1031e))
* add force parameter for cache_with_pickle & using cache when get kaggle leaderboard ([#687](https://github.com/microsoft/RD-Agent/issues/687)) ([c8841e5](https://github.com/microsoft/RD-Agent/commit/c8841e590a925200859acba9fda4a17d4c3aa1c7))
* add retry mechanism for GPU device check in DockerEnv ([#573](https://github.com/microsoft/RD-Agent/issues/573)) ([a780cfb](https://github.com/microsoft/RD-Agent/commit/a780cfb621dc487cc17072bfd4aedd7d581249ab))
* add scores.csv checking in ensemble_test ([#567](https://github.com/microsoft/RD-Agent/issues/567)) ([01808b4](https://github.com/microsoft/RD-Agent/commit/01808b47c314d1daffacc0a65e0ab934a1c41d65))
* add stdout context length setting and improve text shrinking logic ([#559](https://github.com/microsoft/RD-Agent/issues/559)) ([4ac26a6](https://github.com/microsoft/RD-Agent/commit/4ac26a65c1f18f7513480dd562566c8a96298aa7))
* align components' name ([#701](https://github.com/microsoft/RD-Agent/issues/701)) ([295a114](https://github.com/microsoft/RD-Agent/commit/295a1148c53d00b716b2d540573a7f43e7e2d762))
* auto continue small bug ([#598](https://github.com/microsoft/RD-Agent/issues/598)) ([75eaecf](https://github.com/microsoft/RD-Agent/commit/75eaecf36b9f70dfc2d7fedd35836acdb05f89d6))
* avoid try-except in ensemble eval prompts ([#637](https://github.com/microsoft/RD-Agent/issues/637)) ([5c58d6e](https://github.com/microsoft/RD-Agent/commit/5c58d6e524ef848024578033ab6d47bc9b220822))
* avoid warning for missing llama installation when not in use ([#509](https://github.com/microsoft/RD-Agent/issues/509)) ([5ec3422](https://github.com/microsoft/RD-Agent/commit/5ec342224c2c8c4cf591f1eae673e25b14218726))
* change devault to default ([#688](https://github.com/microsoft/RD-Agent/issues/688)) ([7f401cd](https://github.com/microsoft/RD-Agent/commit/7f401cd1c3b333285acf6d6e57654f4b9f0cb6c5))
* change ensemble test ([#622](https://github.com/microsoft/RD-Agent/issues/622)) ([5de3595](https://github.com/microsoft/RD-Agent/commit/5de35953ed0d3e2e1f4dff0e0522f2d6475079ec))
* change summary info of log folder ([#552](https://github.com/microsoft/RD-Agent/issues/552)) ([0eb258d](https://github.com/microsoft/RD-Agent/commit/0eb258d734e9a1280a238b9a6f63eb33047ee0a7))
* clarify an ambiguous explanation ([#705](https://github.com/microsoft/RD-Agent/issues/705)) ([5dbfc68](https://github.com/microsoft/RD-Agent/commit/5dbfc6859cbf6cc31932dae30cf05506108fc871))
* clarify cross_validation ([#644](https://github.com/microsoft/RD-Agent/issues/644)) ([906993e](https://github.com/microsoft/RD-Agent/commit/906993ef6482f88131d1af46f5bc66a77034b549))
* coder prompt & model test text ([#583](https://github.com/microsoft/RD-Agent/issues/583)) ([0a41227](https://github.com/microsoft/RD-Agent/commit/0a41227f267050feaeeb47ddd4d749643eb9f198))
* correct the configuration inheritance relationship ([#671](https://github.com/microsoft/RD-Agent/issues/671)) ([30b1ff8](https://github.com/microsoft/RD-Agent/commit/30b1ff8e1ce59b741e0b81481962063014641c0b))
* default emb model ([#702](https://github.com/microsoft/RD-Agent/issues/702)) ([4329a72](https://github.com/microsoft/RD-Agent/commit/4329a722832a201b3fa6f9d8f9d8d46f78110410))
* direct_exp_gen to json_target_type in DSExpGen class ([#661](https://github.com/microsoft/RD-Agent/issues/661)) ([428b74a](https://github.com/microsoft/RD-Agent/commit/428b74a988157ea864ebb40e828bd9f67589c863))
* docker error will trigger retry and data science runner loop set to 3 ([#602](https://github.com/microsoft/RD-Agent/issues/602)) ([ad785e0](https://github.com/microsoft/RD-Agent/commit/ad785e03d5db05d9191d5e772e184532835a787b))
* ensure expected type ([#593](https://github.com/microsoft/RD-Agent/issues/593)) ([098a9a6](https://github.com/microsoft/RD-Agent/commit/098a9a6618f70fa8dd276b9014b9e7ba9621553b))
* filter empty log traces in ds UI ([#533](https://github.com/microsoft/RD-Agent/issues/533)) ([1a2057c](https://github.com/microsoft/RD-Agent/commit/1a2057c9fc11edc4637f0baaa6dd226eb049c36e))
* fix a bug in cross validation ([#618](https://github.com/microsoft/RD-Agent/issues/618)) ([05a4f10](https://github.com/microsoft/RD-Agent/commit/05a4f101e0b64b860ad03294619b2350004657e8))
* fix a bug in ensemble test script ([#713](https://github.com/microsoft/RD-Agent/issues/713)) ([ad32100](https://github.com/microsoft/RD-Agent/commit/ad321000acbd9291d22fe03a9c60e57c70511c73))
* fix a bug in initial tasks ([#635](https://github.com/microsoft/RD-Agent/issues/635)) ([edb552e](https://github.com/microsoft/RD-Agent/commit/edb552ed283119444f357fbd0b6170b2ad97712a))
* fix a bug in kaggle conf ([#459](https://github.com/microsoft/RD-Agent/issues/459)) ([b4ed32b](https://github.com/microsoft/RD-Agent/commit/b4ed32b17ef07d8557450063765585a48d5fcd32))
* fix a bug in progress_bar filter ([#712](https://github.com/microsoft/RD-Agent/issues/712)) ([ba5a84d](https://github.com/microsoft/RD-Agent/commit/ba5a84dee59c39cc2a8c0d428a82da1f899ce537))
* fix a bug in proposal (add last loop's exception to last task desc) ([#596](https://github.com/microsoft/RD-Agent/issues/596)) ([419186f](https://github.com/microsoft/RD-Agent/commit/419186ffb985fe5a0aa0f7fe59c7a223e355492e))
* fix a bug in threshold score display ([#592](https://github.com/microsoft/RD-Agent/issues/592)) ([0b0a2dc](https://github.com/microsoft/RD-Agent/commit/0b0a2dc512a5560a66464ad49de25d362d0dc17e))
* fix a bug related to model_name in ensemble ([#692](https://github.com/microsoft/RD-Agent/issues/692)) ([c6ce473](https://github.com/microsoft/RD-Agent/commit/c6ce4733f32578298abe0b60f9d82611b793cc09))
* fix a minor bug ([#694](https://github.com/microsoft/RD-Agent/issues/694)) ([1405d8d](https://github.com/microsoft/RD-Agent/commit/1405d8dafd99ecde6f3ba9dd76133d8830d03b47))
* fix an error in model_coder prompt ([#690](https://github.com/microsoft/RD-Agent/issues/690)) ([4528826](https://github.com/microsoft/RD-Agent/commit/452882674e915dbd9e3399c26c70ce5bb86d012c))
* fix combined_factors_df.pkl not loading in docker ([#697](https://github.com/microsoft/RD-Agent/issues/697)) ([3984b99](https://github.com/microsoft/RD-Agent/commit/3984b995aa74318b40de7712e100d4de5cc95b11))
* fix docs build error ([#711](https://github.com/microsoft/RD-Agent/issues/711)) ([c9e1d32](https://github.com/microsoft/RD-Agent/commit/c9e1d32d6b63560350cc7cb799c3a908e2c04e42))
* fix ExtendedSettingsConfigDict does not work ([#660](https://github.com/microsoft/RD-Agent/issues/660)) ([3a877f3](https://github.com/microsoft/RD-Agent/commit/3a877f383b908da8d027560714030b201946bb76))
* fix some bugs (ensemble output, HPO, model tuning) ([#648](https://github.com/microsoft/RD-Agent/issues/648)) ([818ee29](https://github.com/microsoft/RD-Agent/commit/818ee29f8e5d4765b9801463b85b42ee9516ec33))
* fix some bugs in the ensemble component ([#595](https://github.com/microsoft/RD-Agent/issues/595)) ([c0990ab](https://github.com/microsoft/RD-Agent/commit/c0990abb06c73ae062d9a50f50cdfd6d04aded22))
* fix some bugs in workflow unit test ([#624](https://github.com/microsoft/RD-Agent/issues/624)) ([f845dcc](https://github.com/microsoft/RD-Agent/commit/f845dcc0ee1b059b8b32485ad46bb90c7ae0fa78))
* fix some description errors in direct_exp_gen ([#698](https://github.com/microsoft/RD-Agent/issues/698)) ([dfaacb6](https://github.com/microsoft/RD-Agent/commit/dfaacb6d06e5d5f55e950d7177570d1efebf958f))
* fix some minor bugs and add AutoML & cross-validation ([#604](https://github.com/microsoft/RD-Agent/issues/604)) ([18c5ef2](https://github.com/microsoft/RD-Agent/commit/18c5ef268d40efe7bb9ee18aa0d250732bdda6fa))
* fix submission file search and add TODO in env.py ([#544](https://github.com/microsoft/RD-Agent/issues/544)) ([54d930e](https://github.com/microsoft/RD-Agent/commit/54d930e91e629f0fc2f8bdd0d0d62fcad1e99a9c))
* fix task return dict with wrong format ([#558](https://github.com/microsoft/RD-Agent/issues/558)) ([2008244](https://github.com/microsoft/RD-Agent/commit/20082440a249dd0e5a7026c2d98c9de0288dd400))
* fix the errors in the coder and evaluator of the five components ([#576](https://github.com/microsoft/RD-Agent/issues/576)) ([c487f83](https://github.com/microsoft/RD-Agent/commit/c487f835b651cdc40b95bbbe4efcb9a617be9e40))
* handle division by zero in percentage calculations ([#550](https://github.com/microsoft/RD-Agent/issues/550)) ([de16c91](https://github.com/microsoft/RD-Agent/commit/de16c915e1716ef8cee43ce41069ea1a09cf1f24))
* handle invalid regex patterns in filter_progress_bar function ([#579](https://github.com/microsoft/RD-Agent/issues/579)) ([b0daee0](https://github.com/microsoft/RD-Agent/commit/b0daee0d90e193ca1d028e01c31ebf368af89601))
* Handle ValueError when resolving relative path for uri ([#585](https://github.com/microsoft/RD-Agent/issues/585)) ([4c7765a](https://github.com/microsoft/RD-Agent/commit/4c7765a12bda5dcfd9af72b292853d9bc28c5baf))
* include data information in cache key generation ([#566](https://github.com/microsoft/RD-Agent/issues/566)) ([26dda46](https://github.com/microsoft/RD-Agent/commit/26dda4682b7b643c164589057cb568a4d9e55e17))
* keep some txt files ([#557](https://github.com/microsoft/RD-Agent/issues/557)) ([54aba85](https://github.com/microsoft/RD-Agent/commit/54aba851c9fa194e318d37700307df59e06c6c84))
* mle_score save problem ([#674](https://github.com/microsoft/RD-Agent/issues/674)) ([ca2e478](https://github.com/microsoft/RD-Agent/commit/ca2e478cf25c2c8511d5f027e32f8a98fc8e3a07))
* move docker timeout message to __run() ([#620](https://github.com/microsoft/RD-Agent/issues/620)) ([585f4f9](https://github.com/microsoft/RD-Agent/commit/585f4f96e09f70d00eb397c10bf49c09973111df))
* move mlebench check into runner ([#556](https://github.com/microsoft/RD-Agent/issues/556)) ([b0f7965](https://github.com/microsoft/RD-Agent/commit/b0f7965f650638273710302efee2e5da037368a2))
* move next_component_required logic to DSTrace class and accurate implement ([#612](https://github.com/microsoft/RD-Agent/issues/612)) ([c20d311](https://github.com/microsoft/RD-Agent/commit/c20d311792f33b2ccccb466c6ec3155ff8be3213))
* patching weird azure deployment ([#494](https://github.com/microsoft/RD-Agent/issues/494)) ([89c50ae](https://github.com/microsoft/RD-Agent/commit/89c50aee2ec8bfd1cb23767ddf7dcdd023daac8b))
* qlib and other scenario bugs ([#636](https://github.com/microsoft/RD-Agent/issues/636)) ([98de31d](https://github.com/microsoft/RD-Agent/commit/98de31d4e577c8c450c9694f73a755c19af571f7))
* refine prompt to generate the most simple task in init stage ([#546](https://github.com/microsoft/RD-Agent/issues/546)) ([9d6feed](https://github.com/microsoft/RD-Agent/commit/9d6feed28ce034db48482d8d9741ef8c72f4bddc))
* replace API call with build_cls_from_json_with_retry function ([#548](https://github.com/microsoft/RD-Agent/issues/548)) ([eb72a47](https://github.com/microsoft/RD-Agent/commit/eb72a47fbf9c88dacea9691b8d7e92610492d190))
* return 1D embedding if create_embedding receive a string input ([#670](https://github.com/microsoft/RD-Agent/issues/670)) ([4a9c318](https://github.com/microsoft/RD-Agent/commit/4a9c3180ae4a4b043b1b4a89f51ee69cb6843142))
* rich.print error when some control char in output ([#684](https://github.com/microsoft/RD-Agent/issues/684)) ([ec0cb2a](https://github.com/microsoft/RD-Agent/commit/ec0cb2a032824023dcd04a3acc93202471d1f90a))
* Runnable on first complete & Rename method to next_incomplete_component for clarity ([#615](https://github.com/microsoft/RD-Agent/issues/615)) ([93d9f63](https://github.com/microsoft/RD-Agent/commit/93d9f63369a78f78e1a67ab548923bb994d1d3b4))
* runner COSTEER evaluator ([#693](https://github.com/microsoft/RD-Agent/issues/693)) ([6a379ec](https://github.com/microsoft/RD-Agent/commit/6a379ec9b84d4e4944f1e412347aae4f5a93d476))
* save only one mle_score pkl for a running exp ([#675](https://github.com/microsoft/RD-Agent/issues/675)) ([f87ab67](https://github.com/microsoft/RD-Agent/commit/f87ab676b73cce82bd9f997ac779e31c571b53c4))
* Set default value for 'entry' parameter in Env.run method ([#643](https://github.com/microsoft/RD-Agent/issues/643)) ([e50d242](https://github.com/microsoft/RD-Agent/commit/e50d2424b849e4181d6ca02e9cace90236665924))
* sort file name for cache reproduction ([#588](https://github.com/microsoft/RD-Agent/issues/588)) ([7158410](https://github.com/microsoft/RD-Agent/commit/7158410fbfdd84052f9a69cf1e04e09ac07ca598))
* sota comparison logic ([#608](https://github.com/microsoft/RD-Agent/issues/608)) ([3575372](https://github.com/microsoft/RD-Agent/commit/35753722c0800d62855faeab996d513e62cfe7de))
* target json type & round ([#662](https://github.com/microsoft/RD-Agent/issues/662)) ([58cb58f](https://github.com/microsoft/RD-Agent/commit/58cb58f966a1db26f5ea9662a54ba12bc921ee24))
* templates bug ([#456](https://github.com/microsoft/RD-Agent/issues/456)) ([434a868](https://github.com/microsoft/RD-Agent/commit/434a8687eeda77e27b4938fb19694c15858ee446))
* trace summary df showing in dsapp ([#551](https://github.com/microsoft/RD-Agent/issues/551)) ([177096d](https://github.com/microsoft/RD-Agent/commit/177096d55fecb8c7dab9650ef8f5a31024cd4c1c))
* unzip kaggle data ([#464](https://github.com/microsoft/RD-Agent/issues/464)) ([3a9fc8e](https://github.com/microsoft/RD-Agent/commit/3a9fc8e73337d3757267b6f4482499499a1b6792))

## [0.3.0](https://github.com/microsoft/RD-Agent/compare/v0.2.1...v0.3.0) (2024-10-21)


### Features

* add a new template for kaggle ([#289](https://github.com/microsoft/RD-Agent/issues/289)) ([eee3ab5](https://github.com/microsoft/RD-Agent/commit/eee3ab5b25198224826cb7a8a17eab28bd5d1f7d))
* add download submission.csv button for kaggle scenario ([#317](https://github.com/microsoft/RD-Agent/issues/317)) ([dcdcbe4](https://github.com/microsoft/RD-Agent/commit/dcdcbe46b4858bfb133ae3cca056e7f602d5cf63))
* add kaggle command ([#271](https://github.com/microsoft/RD-Agent/issues/271)) ([0938394](https://github.com/microsoft/RD-Agent/commit/0938394b7084ffbf3294d8c23d2d34bf7322ca0b))
* add kaggle tpl: feedback-prize ([#331](https://github.com/microsoft/RD-Agent/issues/331)) ([a288e39](https://github.com/microsoft/RD-Agent/commit/a288e399e6b0beec62729bd7d46b98a55de5ab79))
* add more templates for kaggle ([#291](https://github.com/microsoft/RD-Agent/issues/291)) ([da752ec](https://github.com/microsoft/RD-Agent/commit/da752ec806e6f5f5679bc27ac1c072ed9a319251))
* add normal rag into framework ([#360](https://github.com/microsoft/RD-Agent/issues/360)) ([91b0b1f](https://github.com/microsoft/RD-Agent/commit/91b0b1f66c3c1bf757cb64c4cfbdcaafe59eab74))
* add qlib_factor_strategy ([#307](https://github.com/microsoft/RD-Agent/issues/307)) ([f8f59ff](https://github.com/microsoft/RD-Agent/commit/f8f59ff0a1be4428a68c8c27f220aabad0b6c9f0))
* Add ranking in kaggle scenario ([#401](https://github.com/microsoft/RD-Agent/issues/401)) ([b16b4be](https://github.com/microsoft/RD-Agent/commit/b16b4beb402e0c27dfb39ee9d2a120f1b56d447c))
* Add runtime measurement for each step and loop in RDLoop. ([#281](https://github.com/microsoft/RD-Agent/issues/281)) ([83058c8](https://github.com/microsoft/RD-Agent/commit/83058c864ceeec413dd29bf501030d5a7bd34679))
* add s3e11 kaggle template ([#324](https://github.com/microsoft/RD-Agent/issues/324)) ([8c57524](https://github.com/microsoft/RD-Agent/commit/8c57524bead1c8f655a08763d608eb7a6dd5975e))
* Added RepoAnalyzer to empower auto-summary of a workspace ([#264](https://github.com/microsoft/RD-Agent/issues/264)) ([0bd349a](https://github.com/microsoft/RD-Agent/commit/0bd349af50b9b881ba1774bdeb4d723529ef2aa9))
* Added support for loading and storing RAG in Kaggle scenarios. ([#269](https://github.com/microsoft/RD-Agent/issues/269)) ([c4895de](https://github.com/microsoft/RD-Agent/commit/c4895de83f1ed000e563d42b3468a6bd9e5a4965))
* announce Discord and WeChat ([#367](https://github.com/microsoft/RD-Agent/issues/367)) ([acac507](https://github.com/microsoft/RD-Agent/commit/acac5078a103b71afa6bd6c053b0766a6a7e609d))
* auto submit result after one kaggle RDLoop ([#345](https://github.com/microsoft/RD-Agent/issues/345)) ([ab55d70](https://github.com/microsoft/RD-Agent/commit/ab55d7052b53a928b84dc5d5d0d2999d90ca9056))
* better feedback & evaluation ([#346](https://github.com/microsoft/RD-Agent/issues/346)) ([cc9a8c1](https://github.com/microsoft/RD-Agent/commit/cc9a8c1eab3ca89f8c1e5de4a2bb4e7fcc0cc615))
* Dynamic scenario based on task ([#392](https://github.com/microsoft/RD-Agent/issues/392)) ([665a037](https://github.com/microsoft/RD-Agent/commit/665a037e4fd7326c450e3fa0d0605eea26fd9ef3))
* Factor Implement Search Enhancement ([#294](https://github.com/microsoft/RD-Agent/issues/294)) ([4ecf25f](https://github.com/microsoft/RD-Agent/commit/4ecf25f0acf2389a172b14d3dab20895daf2ab89))
* Feature selection v3 to support all actions  ([#280](https://github.com/microsoft/RD-Agent/issues/280)) ([0047641](https://github.com/microsoft/RD-Agent/commit/00476413fbf00e36e71ab3ccb48d4e766b6ccf4d))
* fix some bugs and add original features' description ([#259](https://github.com/microsoft/RD-Agent/issues/259)) ([1a5f45a](https://github.com/microsoft/RD-Agent/commit/1a5f45a40d821c017bdba14af8c93710707c5ea5))
* get kaggle notebooks & disscussion text for RAG ([#371](https://github.com/microsoft/RD-Agent/issues/371)) ([cead345](https://github.com/microsoft/RD-Agent/commit/cead3450a14bf4b142ac988c27fa098c7656a95c))
* Iceberge competition ([#372](https://github.com/microsoft/RD-Agent/issues/372)) ([c10ea4f](https://github.com/microsoft/RD-Agent/commit/c10ea4f5d4cc56a75b47cf23c7084ee189ba1a25))
* implement isolated model feature selection loop ([#370](https://github.com/microsoft/RD-Agent/issues/370)) ([cf1292d](https://github.com/microsoft/RD-Agent/commit/cf1292de1a0153ca14ea64971e73a1c93f7d89e3))
* Initial version if Graph RAG in KAGGLE scenario ([#301](https://github.com/microsoft/RD-Agent/issues/301)) ([fd3c0fd](https://github.com/microsoft/RD-Agent/commit/fd3c0fd26eff7d3be72fa4f2a234e33b9f796627))
* Integrate RAG into the Kaggle scenarios. ([#262](https://github.com/microsoft/RD-Agent/issues/262)) ([be0e48a](https://github.com/microsoft/RD-Agent/commit/be0e48a7dfbee2b5d2947d09115db5db2e5266f1))
* Kaggle loop update (Feature & Model) ([#241](https://github.com/microsoft/RD-Agent/issues/241)) ([4cf22a6](https://github.com/microsoft/RD-Agent/commit/4cf22a65c964123b4267569ee02c0c7094c54ca4))
* kaggle templates related ([#287](https://github.com/microsoft/RD-Agent/issues/287)) ([785fdc1](https://github.com/microsoft/RD-Agent/commit/785fdc144d16fa8454b7c9d2e53e78fe7f22a29a))
* Model context for tuning and selection ([#284](https://github.com/microsoft/RD-Agent/issues/284)) ([f2831e7](https://github.com/microsoft/RD-Agent/commit/f2831e7442510668b0ca75953b3359894803ef3c))
* Modify FactorRowCountEvaluator and FactorIndexEvaluator to return the ratio ([#328](https://github.com/microsoft/RD-Agent/issues/328)) ([8f43f8e](https://github.com/microsoft/RD-Agent/commit/8f43f8e87a92e05b541e925910608606ec8f6c4b))
* New competition - Optiver ([#356](https://github.com/microsoft/RD-Agent/issues/356)) ([3705efe](https://github.com/microsoft/RD-Agent/commit/3705efe3b923748655a57d76b7a236e54d361831))
* random forest for s3e11 ([#347](https://github.com/microsoft/RD-Agent/issues/347)) ([b57846d](https://github.com/microsoft/RD-Agent/commit/b57846d29314e9a5967945d1b4895f0f48c0f5ce))
* refine the code in model description and fix some bugs in feedback.py ([#288](https://github.com/microsoft/RD-Agent/issues/288)) ([5b124d7](https://github.com/microsoft/RD-Agent/commit/5b124d7372137e4c613eb2749ddcc773922cc7b6))
* refine the template in several Kaggle competitions ([#343](https://github.com/microsoft/RD-Agent/issues/343)) ([034f238](https://github.com/microsoft/RD-Agent/commit/034f238ed5ec351486b21250eabc75114961936c))
* Revise to support better hypothesis proposal  ([#390](https://github.com/microsoft/RD-Agent/issues/390)) ([c55ec0a](https://github.com/microsoft/RD-Agent/commit/c55ec0a0f577bbf7fc6228f7b87d2089ded83b31))
* show workspace in demo ([#348](https://github.com/microsoft/RD-Agent/issues/348)) ([ddf567c](https://github.com/microsoft/RD-Agent/commit/ddf567c551b553788be022e9312c209ef6137d64))
* support Multi output ([#330](https://github.com/microsoft/RD-Agent/issues/330)) ([3d36c45](https://github.com/microsoft/RD-Agent/commit/3d36c452ff0983800e5343834cc69f24a508ea70))
* Supporting COVID-19 competition ([#374](https://github.com/microsoft/RD-Agent/issues/374)) ([a1b63db](https://github.com/microsoft/RD-Agent/commit/a1b63db79600edc9a74ba713c9d0be290214a592))
* supporting Mnist competition ([#375](https://github.com/microsoft/RD-Agent/issues/375)) ([e958a34](https://github.com/microsoft/RD-Agent/commit/e958a34f5632a46ac43bff8e0d07d6ed020fdfc2))
* Supporting Model Specifications ([#319](https://github.com/microsoft/RD-Agent/issues/319)) ([e126471](https://github.com/microsoft/RD-Agent/commit/e1264719e10b76158a91cd0ef331848e7c2de7c7))
* supporting various Kaggle competitions & scenarios for RD-Agent ([#409](https://github.com/microsoft/RD-Agent/issues/409)) ([75eea22](https://github.com/microsoft/RD-Agent/commit/75eea22cc3d4e6f5a94c88cce915e27c507f8c50))
* template for kaggle ([#308](https://github.com/microsoft/RD-Agent/issues/308)) ([ff97cf0](https://github.com/microsoft/RD-Agent/commit/ff97cf0155ab6941e4b5cf7d103575f934b70dc9))
* use auto gen seed when using LLM cache ([#441](https://github.com/microsoft/RD-Agent/issues/441)) ([ca15365](https://github.com/microsoft/RD-Agent/commit/ca15365d23eeb094f42cf3dc8f5269b2f1c42bd3))
* use unified pickle cacher & move llm config into a isolated config ([#424](https://github.com/microsoft/RD-Agent/issues/424)) ([2879ecf](https://github.com/microsoft/RD-Agent/commit/2879ecff816d97688b60909a79c7e568d42608a1))
* xgboost gpu accelerate ([#359](https://github.com/microsoft/RD-Agent/issues/359)) ([56a5b8f](https://github.com/microsoft/RD-Agent/commit/56a5b8f9b2c6726cc64ec5b04b4ce7935d59b572))


### Bug Fixes

* a bug of developer& edit s4e8 template ([#338](https://github.com/microsoft/RD-Agent/issues/338)) ([f12ce72](https://github.com/microsoft/RD-Agent/commit/f12ce726e7de96d478a232a3c27f92439820f8b4))
* actively raised errors aer also considered as negative feedback. ([#268](https://github.com/microsoft/RD-Agent/issues/268)) ([46ec908](https://github.com/microsoft/RD-Agent/commit/46ec908e3594ac5e4cdc4057268e2f8800f5ed1f))
* bug of saving preprocess cache files ([#310](https://github.com/microsoft/RD-Agent/issues/310)) ([5fb0608](https://github.com/microsoft/RD-Agent/commit/5fb0608f39f113cc9807fb1f381284a0bd4da318))
* cache ([#383](https://github.com/microsoft/RD-Agent/issues/383)) ([f2a6e75](https://github.com/microsoft/RD-Agent/commit/f2a6e75b36ca96f7733b9c2a7154ac67bd2d7c6f))
* change css tag of kaggle competition info crawler ([#306](https://github.com/microsoft/RD-Agent/issues/306)) ([1e3d38b](https://github.com/microsoft/RD-Agent/commit/1e3d38bf1ca3654f3a90ff392ecba1dbb4e80224))
* debug dsagent ([#387](https://github.com/microsoft/RD-Agent/issues/387)) ([8fe9511](https://github.com/microsoft/RD-Agent/commit/8fe9511e606ba148c66f384add6ab94857079541))
* eval_method cannot catch run factor error ([#260](https://github.com/microsoft/RD-Agent/issues/260)) ([2aaab31](https://github.com/microsoft/RD-Agent/commit/2aaab317ccb7a0121063bcd85fc36c21c7b8a391))
* fix a bug in competition metric evaluation ([#407](https://github.com/microsoft/RD-Agent/issues/407)) ([94c47d6](https://github.com/microsoft/RD-Agent/commit/94c47d6fd5c3e38fc786a83e6d0d05e8d04498f3))
* fix a bug in mini case ([#389](https://github.com/microsoft/RD-Agent/issues/389)) ([e75bb57](https://github.com/microsoft/RD-Agent/commit/e75bb5746f63933b750406bbd34ee63c5ba76b9f))
* fix a bug in model tuning feedback ([#316](https://github.com/microsoft/RD-Agent/issues/316)) ([8aa088d](https://github.com/microsoft/RD-Agent/commit/8aa088da2dc7525a3970c01d01987246f47d6238))
* fix a bug in scenario.py ([#388](https://github.com/microsoft/RD-Agent/issues/388)) ([999a1eb](https://github.com/microsoft/RD-Agent/commit/999a1eb0eff9088e1b02419db741db4acf8d9ff7))
* fix a bug in the format of the model input ([#327](https://github.com/microsoft/RD-Agent/issues/327)) ([8f0574e](https://github.com/microsoft/RD-Agent/commit/8f0574eaaadb245b8c38e09ad4821306996d926f))
* fix a small bug in cache using module name and function name as unique folder name ([#429](https://github.com/microsoft/RD-Agent/issues/429)) ([4f8134a](https://github.com/microsoft/RD-Agent/commit/4f8134a697d952f7ac824d7ebeec64bbc4545ab3))
* fix a typo ([#362](https://github.com/microsoft/RD-Agent/issues/362)) ([9fafabd](https://github.com/microsoft/RD-Agent/commit/9fafabdf321b818bdd2211a2324d50cd0ebe1c1f))
* fix cache result logic ([#430](https://github.com/microsoft/RD-Agent/issues/430)) ([5e34263](https://github.com/microsoft/RD-Agent/commit/5e342637dcc862679fd0642c6ba9ef048c984845))
* fix command injection ([#421](https://github.com/microsoft/RD-Agent/issues/421)) ([52f30a6](https://github.com/microsoft/RD-Agent/commit/52f30a6184af1295be15e855a80b84bc424fc75d))
* fix json load error ([#386](https://github.com/microsoft/RD-Agent/issues/386)) ([bba55fb](https://github.com/microsoft/RD-Agent/commit/bba55fb48fe105f4847c1b9c476eedc80835f523))
* fix some bugs in feedback.py and refine the prompt ([#292](https://github.com/microsoft/RD-Agent/issues/292)) ([d834052](https://github.com/microsoft/RD-Agent/commit/d8340527f133dcc649d599d90d6402eddd37859e))
* fix some bugs in knowledge base ([#378](https://github.com/microsoft/RD-Agent/issues/378)) ([fa6ff8e](https://github.com/microsoft/RD-Agent/commit/fa6ff8e591cf1847df77d73116649c5623161573))
* fix some bugs in rag ([#399](https://github.com/microsoft/RD-Agent/issues/399)) ([194215c](https://github.com/microsoft/RD-Agent/commit/194215c4559aee5b6ece18d65c95fb30968e2db6))
* fix some bugs in the entire loop ([#274](https://github.com/microsoft/RD-Agent/issues/274)) ([8a564ec](https://github.com/microsoft/RD-Agent/commit/8a564ece1d87b27ee98b76db317935e802468965))
* fix some errors in scenario.py, proposal.py and runner.py and several complex competition scenarios([#365](https://github.com/microsoft/RD-Agent/issues/365)) ([2e383b1](https://github.com/microsoft/RD-Agent/commit/2e383b175d8448a67cb470f4e3ae8977d8ec6b5b))
* improve_execution_time_in_kaggle_loop ([#279](https://github.com/microsoft/RD-Agent/issues/279)) ([4c8f998](https://github.com/microsoft/RD-Agent/commit/4c8f998c76f1e983a5687d2c65d3251750f2a9a0))
* kaggle data mount problem ([#297](https://github.com/microsoft/RD-Agent/issues/297)) ([795df31](https://github.com/microsoft/RD-Agent/commit/795df311e3f93cd2f3fb51ba5698adaf10f6bd62))
* Optiver fixes ([#357](https://github.com/microsoft/RD-Agent/issues/357)) ([b054017](https://github.com/microsoft/RD-Agent/commit/b054017463af0d1784407030f2477d212118f341))
* partial bug in bench ([#368](https://github.com/microsoft/RD-Agent/issues/368)) ([af9808f](https://github.com/microsoft/RD-Agent/commit/af9808f98736a2df07e121c2f6d7bfeb7b7d3581))
* preprocess output format & some mistake in spelling ([#358](https://github.com/microsoft/RD-Agent/issues/358)) ([b8b2cd6](https://github.com/microsoft/RD-Agent/commit/b8b2cd6ccd3b27aa73de847e50899a8a53b71b8f))
* rag save file ([#385](https://github.com/microsoft/RD-Agent/issues/385)) ([1cb01dd](https://github.com/microsoft/RD-Agent/commit/1cb01dd6fe595f2f5fb86487601326611dd1a57a))
* raise error in demo when no Metric in a Loop ([#313](https://github.com/microsoft/RD-Agent/issues/313)) ([e46a78e](https://github.com/microsoft/RD-Agent/commit/e46a78eb69271cb19978aab2f3b976c2870ca082))
* refactor Bench ([#302](https://github.com/microsoft/RD-Agent/issues/302)) ([78a87f6](https://github.com/microsoft/RD-Agent/commit/78a87f624780ff67c0fa995ae4692678a120f99c))
* refine some codes ([#353](https://github.com/microsoft/RD-Agent/issues/353)) ([866c2e6](https://github.com/microsoft/RD-Agent/commit/866c2e63ffa3876a3d16ad37f96da41d0558b714))
* refine the prompt ([#286](https://github.com/microsoft/RD-Agent/issues/286)) ([77966c4](https://github.com/microsoft/RD-Agent/commit/77966c4f5e9f492c437c5b4b78d89c0f875ef0d8))
* refine the ucb algorithm ([#406](https://github.com/microsoft/RD-Agent/issues/406)) ([14f7d97](https://github.com/microsoft/RD-Agent/commit/14f7d976e03c92d6e727524e0cdad8a03b585016))
* revert model and make SOTA model available to COSTEER ([#351](https://github.com/microsoft/RD-Agent/issues/351)) ([3b7437b](https://github.com/microsoft/RD-Agent/commit/3b7437b87e685188259779cd85a78a0b592de9de))
* stop using markup in docker env print ([#336](https://github.com/microsoft/RD-Agent/issues/336)) ([3009889](https://github.com/microsoft/RD-Agent/commit/3009889b5e2605b5427c76f3084e0e58026bb5ae))
* support seed and fix absolute path ([#278](https://github.com/microsoft/RD-Agent/issues/278)) ([26352e1](https://github.com/microsoft/RD-Agent/commit/26352e13121cad5be95c0de78bb9f5dda4330614))
* template for kaggle foreset & s4e9 ([#334](https://github.com/microsoft/RD-Agent/issues/334)) ([2393a41](https://github.com/microsoft/RD-Agent/commit/2393a41e7237615ced2c3fdd5c49308236b9f276))
* test kaggle method ([#296](https://github.com/microsoft/RD-Agent/issues/296)) ([91a6196](https://github.com/microsoft/RD-Agent/commit/91a619618be1d7db660ea2b413a78dfaba9417a1))
* update code to fix a small bug in model cache md5 hash ([#303](https://github.com/microsoft/RD-Agent/issues/303)) ([b00e4dc](https://github.com/microsoft/RD-Agent/commit/b00e4dc2eff5b16029a2a12a6589eadac5cfd148))
* update new feature engineering code format ([#272](https://github.com/microsoft/RD-Agent/issues/272)) ([7850b80](https://github.com/microsoft/RD-Agent/commit/7850b8006a7c89d22629b345b4f361b0f35bc60d))
* Update prompts.yaml to constrain only one model type   ([#341](https://github.com/microsoft/RD-Agent/issues/341)) ([5b5dfee](https://github.com/microsoft/RD-Agent/commit/5b5dfeefbc7eb9dcbd9923544005c5d281262c03))
* Update runner.py to fix a small bug ([#282](https://github.com/microsoft/RD-Agent/issues/282)) ([8aef3ab](https://github.com/microsoft/RD-Agent/commit/8aef3abcecd6002bd4bfeedcbe2c786d8bbfe2be))
* Use fixed file name in model costeer & fixing cache ([#311](https://github.com/microsoft/RD-Agent/issues/311)) ([1f910a5](https://github.com/microsoft/RD-Agent/commit/1f910a5248bc576895ed66c2f7b2c3e046a2bc28))


### Performance Improvements

* some small upgrade to factor costeer to improve the performance ([#420](https://github.com/microsoft/RD-Agent/issues/420)) ([9eb931f](https://github.com/microsoft/RD-Agent/commit/9eb931ffd971f252380dbd33ad1db259a4f229fd))


### Reverts

* Revert feat: Factor Implement Search Enhancement ([#294](https://github.com/microsoft/RD-Agent/issues/294)) ([#305](https://github.com/microsoft/RD-Agent/issues/305)) ([f663cf4](https://github.com/microsoft/RD-Agent/commit/f663cf42a2f75cd52aef1c6b18be7c27f0641fed))

## [0.2.1](https://github.com/microsoft/RD-Agent/compare/v0.2.0...v0.2.1) (2024-09-10)


### Bug Fixes

* default model value in config ([#256](https://github.com/microsoft/RD-Agent/issues/256)) ([c097585](https://github.com/microsoft/RD-Agent/commit/c097585f631f401c2c0966f6ad4c17286924f011))
* fix_dotenv_error ([#257](https://github.com/microsoft/RD-Agent/issues/257)) ([923063c](https://github.com/microsoft/RD-Agent/commit/923063c1fd957c4ed42e97272c72b5e9545451dc))
* readme ([#248](https://github.com/microsoft/RD-Agent/issues/248)) ([8cede22](https://github.com/microsoft/RD-Agent/commit/8cede2209922876490148459e1134da828e1fda0))

## [0.2.0](https://github.com/microsoft/RD-Agent/compare/v0.1.0...v0.2.0) (2024-09-07)


### Features

* add collect info ([#233](https://github.com/microsoft/RD-Agent/issues/233)) ([89f4af9](https://github.com/microsoft/RD-Agent/commit/89f4af90fb4d95a0689bf9efc8ffd9326469c0aa))
* add cross validation for kaggle scenario ([#236](https://github.com/microsoft/RD-Agent/issues/236)) ([e0b03ba](https://github.com/microsoft/RD-Agent/commit/e0b03ba6b5c3d9aa552b99d470e106d4e348e64d))
* add progress status for docker env ([#215](https://github.com/microsoft/RD-Agent/issues/215)) ([538d4ef](https://github.com/microsoft/RD-Agent/commit/538d4ef2e52de795b90d3f75b2e1e877ab85c18d))
* Added loop code for Kaggle scene. ([#211](https://github.com/microsoft/RD-Agent/issues/211)) ([975c327](https://github.com/microsoft/RD-Agent/commit/975c32715e51aec6b49537401f5fc59115e04a01))
* Demo display effect and usage ([#162](https://github.com/microsoft/RD-Agent/issues/162)) ([8cf122a](https://github.com/microsoft/RD-Agent/commit/8cf122a0155f434fa4477ae7a6d616b5caecd3e0))
* piloting of the framework ([#227](https://github.com/microsoft/RD-Agent/issues/227)) ([e9b103e](https://github.com/microsoft/RD-Agent/commit/e9b103e684fdd2b98cd1a89971a3fce2d6e884a1))
* support more models for kaggle scenario ([#223](https://github.com/microsoft/RD-Agent/issues/223)) ([e3a9659](https://github.com/microsoft/RD-Agent/commit/e3a96598c0720fe092ec86d7ca8c195c7d6bcc72))
* update model_experiment.py to support basic EDA ([#220](https://github.com/microsoft/RD-Agent/issues/220)) ([bf2684c](https://github.com/microsoft/RD-Agent/commit/bf2684c4d55ab8e1048ac0291695475ad53b0cd6))


### Bug Fixes

* fix some bugs in llm calling ([#217](https://github.com/microsoft/RD-Agent/issues/217)) ([7b010f8](https://github.com/microsoft/RD-Agent/commit/7b010f8b5940aba65a58f1d78192aa80bcd0e654))
* package dependency. ([#234](https://github.com/microsoft/RD-Agent/issues/234)) ([46be295](https://github.com/microsoft/RD-Agent/commit/46be2952952af534fd8d98a656c704c688d7cbdd))
* remove useless line ([#177](https://github.com/microsoft/RD-Agent/issues/177)) ([64e9a8e](https://github.com/microsoft/RD-Agent/commit/64e9a8e39a2072a962111db18f5b9565df5b0176))

## [0.1.0](https://github.com/microsoft/RD-Agent/compare/v0.0.1...v0.1.0) (2024-08-09)


### Features

* add entry for rdagent. ([#187](https://github.com/microsoft/RD-Agent/issues/187)) ([121b6d9](https://github.com/microsoft/RD-Agent/commit/121b6d98de38cd03be30cbee47b40baf39a2b60b))
* change ui entry ([#197](https://github.com/microsoft/RD-Agent/issues/197)) ([fa5d335](https://github.com/microsoft/RD-Agent/commit/fa5d3354d22240888f4fc4007d9834f7424632aa))
* remove pdfs and enable online pdf readings ([#183](https://github.com/microsoft/RD-Agent/issues/183)) ([18c0501](https://github.com/microsoft/RD-Agent/commit/18c05016a23d694c7b12759cf1322562dcffc56a))


### Bug Fixes

* Fix a fail href in readme ([#189](https://github.com/microsoft/RD-Agent/issues/189)) ([1b89218](https://github.com/microsoft/RD-Agent/commit/1b89218f6bc697494f4a1b8a42ad18963002714f))
* fix quick start problem ([#191](https://github.com/microsoft/RD-Agent/issues/191)) ([44f61bf](https://github.com/microsoft/RD-Agent/commit/44f61bfa1058a8efb59ca48b7f1417765aeea33e))
* update command line in readme.md ([#192](https://github.com/microsoft/RD-Agent/issues/192)) ([9c45d24](https://github.com/microsoft/RD-Agent/commit/9c45d24a192da02f7d9765cb001097da1bc36c61))

## 0.0.1 (2024-08-08)


### Features

* Add description for scenario experiments. ([#174](https://github.com/microsoft/RD-Agent/issues/174)) ([fbd8c6d](https://github.com/microsoft/RD-Agent/commit/fbd8c6d87e1424c08997103b8e8fbf264858c4ed))
* Added QlibFactorFromReportScenario and improved the report-factor loop. ([#161](https://github.com/microsoft/RD-Agent/issues/161)) ([882c79b](https://github.com/microsoft/RD-Agent/commit/882c79bf11583980e646b130f71cfa20201ffc7b))
* filter feature which is high correlation to former implemented features ([#145](https://github.com/microsoft/RD-Agent/issues/145)) ([e818326](https://github.com/microsoft/RD-Agent/commit/e818326422740e04a4863f7c3c18744dde2ad98f))
* Remove redundant 'key steps' section in frontend scene display. ([#169](https://github.com/microsoft/RD-Agent/issues/169)) ([e767005](https://github.com/microsoft/RD-Agent/commit/e76700513bee29232c93b97414419df330d9be8d))
* streamlit webapp demo for different scenarios ([#135](https://github.com/microsoft/RD-Agent/issues/135)) ([d8da7db](https://github.com/microsoft/RD-Agent/commit/d8da7db865e6653fc4740efee9a843b69bd79699))
* Uploaded Documentation, Updated Prompts & Some Code for model demo ([#144](https://github.com/microsoft/RD-Agent/issues/144)) ([529f935](https://github.com/microsoft/RD-Agent/commit/529f935aa98623f0dc1dda29eecee3ef738dd446))


### Bug Fixes

* Add framework handling for task coding failure. ([#176](https://github.com/microsoft/RD-Agent/issues/176)) ([5e14fa5](https://github.com/microsoft/RD-Agent/commit/5e14fa54a9dd30a94aebe2643b8c9a3b85517a11))
* Comprehensive update to factor extraction. ([#143](https://github.com/microsoft/RD-Agent/issues/143)) ([b5ea040](https://github.com/microsoft/RD-Agent/commit/b5ea04019fd5fa15c0f8b9a7e4f18f490f7057d4))
* first round app folder cleaning ([#166](https://github.com/microsoft/RD-Agent/issues/166)) ([6a5a750](https://github.com/microsoft/RD-Agent/commit/6a5a75021912927deb5e8e4c7ad3ec4b51bfc788))
* fix pickle problem ([#140](https://github.com/microsoft/RD-Agent/issues/140)) ([7ee4258](https://github.com/microsoft/RD-Agent/commit/7ee42587b60d94417f34332cee395cf210dc8a0e))
* fix release CI ([#165](https://github.com/microsoft/RD-Agent/issues/165)) ([85d6a5e](https://github.com/microsoft/RD-Agent/commit/85d6a5ed91113fda34ae079b23c89aa24acd2cb2))
* fix release CI error ([#160](https://github.com/microsoft/RD-Agent/issues/160)) ([1c9f8ef](https://github.com/microsoft/RD-Agent/commit/1c9f8ef287961731944acc9008496b4dddeddca7))
* fix several bugs in data mining scenario ([#147](https://github.com/microsoft/RD-Agent/issues/147)) ([b233380](https://github.com/microsoft/RD-Agent/commit/b233380e2c66fb030db39424f0f040c86e37f5c4))
* fix some small bugs in report-factor loop ([#152](https://github.com/microsoft/RD-Agent/issues/152)) ([a79f9f9](https://github.com/microsoft/RD-Agent/commit/a79f9f93406aff6305a76e6a6abd3852642e4c62))
* fix_release_ci_error ([#150](https://github.com/microsoft/RD-Agent/issues/150)) ([4f82e99](https://github.com/microsoft/RD-Agent/commit/4f82e9960a2638af9d831581185ddd3bac5711fc))
* Fixed some bugs introduced during refactoring. ([#167](https://github.com/microsoft/RD-Agent/issues/167)) ([f8f1445](https://github.com/microsoft/RD-Agent/commit/f8f1445283fb89aefeb2918243c35a219a51a56c))
* optimize some prompts in factor loop. ([#158](https://github.com/microsoft/RD-Agent/issues/158)) ([c2c1330](https://github.com/microsoft/RD-Agent/commit/c2c13300b9ad315a663ec2d0eada414e56c6f54f))


### Miscellaneous Chores

* release 0.0.1 ([1feacd3](https://github.com/microsoft/RD-Agent/commit/1feacd39b21193de11e9bbecf880ddf96d7c261c))
