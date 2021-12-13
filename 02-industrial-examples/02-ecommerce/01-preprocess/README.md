# 웹사이트는 어떻게 나에게 맞는 상품을 추천하는가?
이번 실습에서는 넷플릭스 데이터를 살펴보고 인공지능 모델을 학습하는 과정을 수행해 보겠습니다.

유튜브 영상을 보고 있던 중 나도 모르게 새로운 영상을 추천받으며 보고 있지 않나요?

웹상에서 운영되는 영상, 쇼핑, 광고 기업들은 여러분에게 새로운 상품을 보여 주기 위하여 인공지능을 활용한 추천 알고리즘을 사용하고 있습니다.

### 데이터 구조 확인
2019년까지 업로드된 넷플릭스 데이터가 아래와 같이 주어졌습니다.

![image](https://user-images.githubusercontent.com/61646760/145755616-87344949-fdb0-4c66-a547-a694315c4263.png)

데이터에는 **6234개의 콘텐츠에 대하여 12가지 변수에 대한 값**들이 저장되어 있습니다.

![image](https://user-images.githubusercontent.com/61646760/145755734-fda650be-80e8-4dea-8a21-379536fa95d5.png)

### 데이터 학습
미리 설정된 학습 알고리즘을 활용해 이 데이터로 학습을 진행해 보도록 하겠습니다.

### 지시 사항
1. 아래 코드를 사용하여 인공지능 모델 학습을 수행해 보세요.
    - `ma.preprocess()`
2. `실행` 버튼을 눌러 결과를 확인해 보세요.
3. `제출` 버튼을 눌러 학습을 수행하는 코드를 올바르게 작성하였는지 확인해 보세요.

## 답안
```
import machine as ma

def main():
    
    """
    지시사항 1번. 인공지능 모델 학습을 수행해 보세요.
    """
    netflix_overall, cosine_sim, indices = ma.preprocess()


if __name__ == "__main__":
    main()
```

### 실행 결과
```
데이터 읽는 중...
읽어 온 데이터를 출력합니다.
       show_id     type                                              title  \
0     81145628    Movie            Norm of the North: King Sized Adventure   
1     80117401    Movie                         Jandino: Whatever it Takes   
2     70234439  TV Show                                 Transformers Prime   
3     80058654  TV Show                   Transformers: Robots in Disguise   
4     80125979    Movie                                       #realityhigh   
5     80163890  TV Show                                            Apaches   
6     70304989    Movie                                           Automata   
7     80164077    Movie                 Fabrizio Copano: Solo pienso en mi   
8     80117902  TV Show                                       Fire Chasers   
9     70304990    Movie                                        Good People   
10    80169755    Movie                        Joaquín Reyes: Una y no más   
11    70299204    Movie                            Kidnapping Mr. Heineken   
12    80182480    Movie                           Krish Trish and Baltiboy   
13    80182483    Movie           Krish Trish and Baltiboy: Battle of Wits   
14    80182596    Movie     Krish Trish and Baltiboy: Best Friends Forever   
15    80182482    Movie          Krish Trish and Baltiboy: Comics of India   
16    80182597    Movie  Krish Trish and Baltiboy: Oversmartness Never ...   
17    80182481    Movie                  Krish Trish and Baltiboy: Part II   
18    80182621    Movie       Krish Trish and Baltiboy: The Greatest Trick   
19    80057969    Movie                                               Love   
20    80060297    Movie                                  Manhattan Romance   
21    80046728    Movie                                        Moonwalkers   
22    80046727    Movie                                     Rolling Papers   
23    70304988    Movie                                 Stonehearst Asylum   
24    80057700    Movie                                         The Runner   
25    80045922    Movie                                            6 Years   
26    80244601  TV Show                                    Castle of Stars   
27    80203094    Movie                                        City of Joy   
28    80190843  TV Show                                     First and Last   
29    70241607    Movie                                          Laddaland   
...        ...      ...                                                ...   
6204  80091341  TV Show                                             Cuckoo   
6205  80036747  TV Show                        Pororo - The Little Penguin   
6206  80173174  TV Show                                          Samantha!   
6207  80190407  TV Show                                  Murderous Affairs   
6208  70227189  TV Show                                          Lost Girl   
6209  70264078  TV Show                                          Mr. Young   
6210  80239700  TV Show                                        Psiconautas   
6211  80231523  TV Show                                The Minimighty Kids   
6212  80126877  TV Show                                            Filinta   
6213  80126599  TV Show                                   Leyla and Mecnun   
6214  80049872  TV Show                                            Chelsea   
6215  80066227  TV Show                                Crazy Ex-Girlfriend   
6216  80108373  TV Show                   The Magic School Bus Rides Again   
6217  70196145  TV Show                                           New Girl   
6218  80162994  TV Show                            Talking Tom and Friends   
6219  80186475  TV Show                                 Pokémon the Series   
6220  70272742  TV Show                                        Justin Time   
6221  80067942  TV Show            Terrace House: Boys & Girls in the City   
6222  70136122  TV Show                                              Weeds   
6223  70204989  TV Show                                    Gunslinger Girl   
6224  70304979  TV Show                    Anthony Bourdain: Parts Unknown   
6225  70153412  TV Show                                            Frasier   
6226  70243132  TV Show                                La Familia P. Luche   
6227  80005756  TV Show                       The Adventures of Figaro Pho   
6228  80159925  TV Show                                           Kikoriki   
6229  80000063  TV Show                                       Red vs. Blue   
6230  70286564  TV Show                                              Maron   
6231  80116008    Movie             Little Baby Bum: Nursery Rhyme Friends   
6232  70281022  TV Show        A Young Doctor's Notebook and Other Stories   
6233  70153404  TV Show                                            Friends   

                             director  \
0            Richard Finn, Tim Maltby   
1                                 NaN   
2                                 NaN   
3                                 NaN   
4                    Fernando Lebrija   
5                                 NaN   
6                         Gabe Ibáñez   
7     Rodrigo Toro, Francisco Schultz   
8                                 NaN   
9                   Henrik Ruben Genz   
10              José Miguel Contreras   
11                   Daniel Alfredson   
12                                NaN   
13        Munjal Shroff, Tilak Shetty   
14        Munjal Shroff, Tilak Shetty   
15                       Tilak Shetty   
16                       Tilak Shetty   
17                                NaN   
18        Munjal Shroff, Tilak Shetty   
19                         Gaspar Noé   
20                        Tom O'Brien   
21             Antoine Bardou-Jacquet   
22                      Mitch Dickman   
23                      Brad Anderson   
24                       Austin Stark   
25                      Hannah Fidell   
26                                NaN   
27                    Madeleine Gavin   
28                                NaN   
29                   Sopon Sukdapisit   
...                               ...   
6204                              NaN   
6205                              NaN   
6206                              NaN   
6207                              NaN   
6208                              NaN   
6209                              NaN   
6210                              NaN   
6211                              NaN   
6212                              NaN   
6213                        Onur Ünlü   
6214                              NaN   
6215                              NaN   
6216                              NaN   
6217                              NaN   
6218                              NaN   
6219                              NaN   
6220                              NaN   
6221                              NaN   
6222                              NaN   
6223                              NaN   
6224                              NaN   
6225                              NaN   
6226                              NaN   
6227                              NaN   
6228                              NaN   
6229                              NaN   
6230                              NaN   
6231                              NaN   
6232                              NaN   
6233                              NaN   

                                                   cast  \
0     Alan Marriott, Andrew Toth, Brian Dobson, Cole...   
1                                      Jandino Asporaat   
2     Peter Cullen, Sumalee Montano, Frank Welker, J...   
3     Will Friedle, Darren Criss, Constance Zimmer, ...   
4     Nesta Cooper, Kate Walsh, John Michael Higgins...   
5     Alberto Ammann, Eloy Azorín, Verónica Echegui,...   
6     Antonio Banderas, Dylan McDermott, Melanie Gri...   
7                                       Fabrizio Copano   
8                                                   NaN   
9     James Franco, Kate Hudson, Tom Wilkinson, Omar...   
10                                        Joaquín Reyes   
11    Jim Sturgess, Sam Worthington, Ryan Kwanten, A...   
12    Damandeep Singh Baggan, Smita Malhotra, Baba S...   
13    Damandeep Singh Baggan, Smita Malhotra, Baba S...   
14    Damandeep Singh Baggan, Smita Malhotra, Deepak...   
15    Damandeep Singh Baggan, Smita Malhotra, Baba S...   
16        Rishi Gambhir, Smita Malhotra, Deepak Chachra   
17    Damandeep Singh Baggan, Smita Malhotra, Baba S...   
18    Damandeep Singh Baggan, Smita Malhotra, Baba S...   
19    Karl Glusman, Klara Kristin, Aomi Muyock, Ugo ...   
20    Tom O'Brien, Katherine Waterston, Caitlin Fitz...   
21    Ron Perlman, Rupert Grint, Robert Sheehan, Ste...   
22                                                  NaN   
23    Kate Beckinsale, Jim Sturgess, David Thewlis, ...   
24    Nicolas Cage, Sarah Paulson, Connie Nielsen, W...   
25    Taissa Farmiga, Ben Rosenfield, Lindsay Burdge...   
26    Chaiyapol Pupart, Jintanutda Lummakanon, Worra...   
27                                                  NaN   
28                                                  NaN   
29    Saharat Sangkapreecha, Pok Piyatida Woramusik,...   
...                                                 ...   
6204  Andy Samberg, Taylor Lautner, Greg Davies, Hel...   
6205                                                NaN   
6206  Emmanuelle Araújo, Douglas Silva, Sabrina Nona...   
6207                                                NaN   
6208  Anna Silk, Kris Holden-Ried, Ksenia Solo, Rich...   
6209  Brendan Meyer, Matreya Fedor, Gig Morton, Kurt...   
6210  Guillermo Toledo, Gabriel Goity, Florencia Peñ...   
6211                                                NaN   
6212  Onur Tuna, Serhat Tutumluer, Mehmet Özgür, Naz...   
6213  Ali Atay, Melis Birkan, Serkan Keskin, Ahmet M...   
6214                                                NaN   
6215  Rachel Bloom, Vincent Rodriguez III, Santino F...   
6216  Kate McKinnon, Miles Koseleci-Vieira, Mikaela ...   
6217  Zooey Deschanel, Jake Johnson, Max Greenfield,...   
6218  Colin Hanks, Tom Kenny, James Adomian, Lisa Sc...   
6219  Sarah Natochenny, Laurie Hymes, Jessica Paquet...   
6220            Gage Munroe, Scott McCord, Jenna Warren   
6221  You, Reina Triendl, Ryota Yamasato, Yoshimi To...   
6222  Mary-Louise Parker, Hunter Parrish, Alexander ...   
6223  Yuuka Nanri, Kanako Mitsuhashi, Eri Sendai, Am...   
6224                                   Anthony Bourdain   
6225  Kelsey Grammer, Jane Leeves, David Hyde Pierce...   
6226  Eugenio Derbez, Consuelo Duval, Luis Manuel Áv...   
6227  Luke Jurevicius, Craig Behenna, Charlotte Haml...   
6228                                      Igor Dmitriev   
6229  Burnie Burns, Jason Saldaña, Gustavo Sorola, G...   
6230  Marc Maron, Judd Hirsch, Josh Brener, Nora Zeh...   
6231                                                NaN   
6232  Daniel Radcliffe, Jon Hamm, Adam Godley, Chris...   
6233  Jennifer Aniston, Courteney Cox, Lisa Kudrow, ...   

                                                country         date_added  \
0              United States, India, South Korea, China  September 9, 2019   
1                                        United Kingdom  September 9, 2016   
2                                         United States  September 8, 2018   
3                                         United States  September 8, 2018   
4                                         United States  September 8, 2017   
5                                                 Spain  September 8, 2017   
6                Bulgaria, United States, Spain, Canada  September 8, 2017   
7                                                 Chile  September 8, 2017   
8                                         United States  September 8, 2017   
9        United States, United Kingdom, Denmark, Sweden  September 8, 2017   
10                                                  NaN  September 8, 2017   
11    Netherlands, Belgium, United Kingdom, United S...  September 8, 2017   
12                                                  NaN  September 8, 2017   
13                                                  NaN  September 8, 2017   
14                                                  NaN  September 8, 2017   
15                                                  NaN  September 8, 2017   
16                                                  NaN  September 8, 2017   
17                                                  NaN  September 8, 2017   
18                                                  NaN  September 8, 2017   
19                                      France, Belgium  September 8, 2017   
20                                        United States  September 8, 2017   
21                                      France, Belgium  September 8, 2017   
22                               United States, Uruguay  September 8, 2017   
23                                        United States  September 8, 2017   
24                                        United States  September 8, 2017   
25                                        United States  September 8, 2015   
26                                                  NaN  September 7, 2018   
27                                       United States,  September 7, 2018   
28                                                  NaN  September 7, 2018   
29                                             Thailand  September 7, 2018   
...                                                 ...                ...   
6204                                     United Kingdom     April 19, 2019   
6205                                        South Korea     April 19, 2019   
6206                                             Brazil     April 19, 2019   
6207                                      United States     April 17, 2018   
6208                                             Canada     April 17, 2016   
6209                                             Canada     April 16, 2019   
6210                                          Argentina     April 15, 2018   
6211                                             France     April 15, 2018   
6212                                             Turkey     April 15, 2017   
6213                                             Turkey     April 15, 2017   
6214                                      United States     April 14, 2017   
6215                                      United States     April 13, 2019   
6216                                      United States     April 13, 2018   
6217                                      United States     April 11, 2019   
6218                          Cyprus, Austria, Thailand     April 10, 2019   
6219                                              Japan      April 1, 2019   
6220                                             Canada      April 1, 2016   
6221                                              Japan      April 1, 2016   
6222                                      United States      April 1, 2014   
6223                                              Japan                NaN   
6224                                      United States                NaN   
6225                                      United States                NaN   
6226                                      United States                NaN   
6227                                          Australia                NaN   
6228                                                NaN                NaN   
6229                                      United States                NaN   
6230                                      United States                NaN   
6231                                                NaN                NaN   
6232                                     United Kingdom                NaN   
6233                                      United States                NaN   

      release_year    rating    duration  \
0             2019     TV-PG      90 min   
1             2016     TV-MA      94 min   
2             2013  TV-Y7-FV    1 Season   
3             2016     TV-Y7    1 Season   
4             2017     TV-14      99 min   
5             2016     TV-MA    1 Season   
6             2014         R     110 min   
7             2017     TV-MA      60 min   
8             2017     TV-MA    1 Season   
9             2014         R      90 min   
10            2017     TV-MA      78 min   
11            2015         R      95 min   
12            2009     TV-Y7      58 min   
13            2013     TV-Y7      62 min   
14            2016      TV-Y      65 min   
15            2012     TV-Y7      61 min   
16            2017     TV-Y7      65 min   
17            2010     TV-Y7      58 min   
18            2013     TV-Y7      60 min   
19            2015        NR     135 min   
20            2014     TV-14      98 min   
21            2015         R      96 min   
22            2015     TV-MA      79 min   
23            2014     PG-13     113 min   
24            2015         R      90 min   
25            2015        NR      80 min   
26            2015     TV-14    1 Season   
27            2018     TV-MA      77 min   
28            2018     TV-MA    1 Season   
29            2011     TV-MA     112 min   
...            ...       ...         ...   
6204          2019     TV-14   5 Seasons   
6205          2013      TV-Y   2 Seasons   
6206          2019     TV-MA   2 Seasons   
6207          2017     TV-14   3 Seasons   
6208          2015     TV-14   5 Seasons   
6209          2013      TV-G   2 Seasons   
6210          2016     TV-MA   2 Seasons   
6211          2012      TV-G   2 Seasons   
6212          2015     TV-14   2 Seasons   
6213          2014     TV-PG   3 Seasons   
6214          2017     TV-MA   2 Seasons   
6215          2019     TV-14   4 Seasons   
6216          2018      TV-Y   2 Seasons   
6217          2017     TV-14   7 Seasons   
6218          2017      TV-G   2 Seasons   
6219          2019  TV-Y7-FV   2 Seasons   
6220          2012      TV-Y   2 Seasons   
6221          2016     TV-14   2 Seasons   
6222          2012     TV-MA   8 Seasons   
6223          2008     TV-14   2 Seasons   
6224          2018     TV-PG   5 Seasons   
6225          2003     TV-PG  11 Seasons   
6226          2012     TV-14   3 Seasons   
6227          2015     TV-Y7   2 Seasons   
6228          2010      TV-Y   2 Seasons   
6229          2015        NR  13 Seasons   
6230          2016     TV-MA   4 Seasons   
6231          2016       NaN      60 min   
6232          2013     TV-MA   2 Seasons   
6233          2003     TV-14  10 Seasons   

                                              listed_in  \
0                    Children & Family Movies, Comedies   
1                                       Stand-Up Comedy   
2                                              Kids' TV   
3                                              Kids' TV   
4                                              Comedies   
5     Crime TV Shows, International TV Shows, Spanis...   
6     International Movies, Sci-Fi & Fantasy, Thrillers   
7                                       Stand-Up Comedy   
8                       Docuseries, Science & Nature TV   
9                         Action & Adventure, Thrillers   
10                                      Stand-Up Comedy   
11     Action & Adventure, Dramas, International Movies   
12                             Children & Family Movies   
13                             Children & Family Movies   
14                             Children & Family Movies   
15                             Children & Family Movies   
16                             Children & Family Movies   
17                             Children & Family Movies   
18                             Children & Family Movies   
19              Cult Movies, Dramas, Independent Movies   
20        Comedies, Independent Movies, Romantic Movies   
21    Action & Adventure, Comedies, International Mo...   
22                                        Documentaries   
23                             Horror Movies, Thrillers   
24                           Dramas, Independent Movies   
25          Dramas, Independent Movies, Romantic Movies   
26    International TV Shows, Romantic TV Shows, TV ...   
27                                        Documentaries   
28                                           Docuseries   
29                  Horror Movies, International Movies   
...                                                 ...   
6204  British TV Shows, International TV Shows, TV C...   
6205                          Kids' TV, Korean TV Shows   
6206                International TV Shows, TV Comedies   
6207                         Crime TV Shows, Docuseries   
6208                 TV Dramas, TV Horror, TV Mysteries   
6209                              Kids' TV, TV Comedies   
6210  International TV Shows, Spanish-Language TV Sh...   
6211                              Kids' TV, TV Comedies   
6212  Crime TV Shows, International TV Shows, TV Act...   
6213  International TV Shows, Romantic TV Shows, TV ...   
6214          Stand-Up Comedy & Talk Shows, TV Comedies   
6215                     Romantic TV Shows, TV Comedies   
6216                                           Kids' TV   
6217                     Romantic TV Shows, TV Comedies   
6218                              Kids' TV, TV Comedies   
6219                             Anime Series, Kids' TV   
6220                                           Kids' TV   
6221                 International TV Shows, Reality TV   
6222                             TV Comedies, TV Dramas   
6223                       Anime Series, Crime TV Shows   
6224                                         Docuseries   
6225                     Classic & Cult TV, TV Comedies   
6226  International TV Shows, Spanish-Language TV Sh...   
6227                              Kids' TV, TV Comedies   
6228                                           Kids' TV   
6229  TV Action & Adventure, TV Comedies, TV Sci-Fi ...   
6230                                        TV Comedies   
6231                                             Movies   
6232           British TV Shows, TV Comedies, TV Dramas   
6233                     Classic & Cult TV, TV Comedies   

                                            description  
0     Before planning an awesome wedding for his gra...  
1     Jandino Asporaat riffs on the challenges of ra...  
2     With the help of three human allies, the Autob...  
3     When a prison ship crash unleashes hundreds of...  
4     When nerdy high schooler Dani finally attracts...  
5     A young journalist is forced into a life of cr...  
6     In a dystopian future, an insurance adjuster f...  
7     Fabrizio Copano takes audience participation t...  
8     As California's 2016 fire season rages, brave ...  
9     A struggling couple can't believe their luck w...  
10    Comedian and celebrity impersonator Joaquín Re...  
11    When beer magnate Alfred "Freddy" Heineken is ...  
12    A team of minstrels, including a monkey, cat a...  
13    An artisan is cheated of his payment, a lion o...  
14    A cat, monkey and donkey team up to narrate fo...  
15    In three comic-strip-style tales, a boy tries ...  
16    A cat, monkey and donkey learn the consequence...  
17    Animal minstrels narrate stories about a monke...  
18    The consequences of trickery are explored in s...  
19    A man in an unsatisfying marriage recalls the ...  
20    A filmmaker working on a documentary about lov...  
21    A brain-addled war vet, a failing band manager...  
22    As the newspaper industry takes a hit, The Den...  
23    In 1899, a young doctor arrives at an asylum f...  
24    A New Orleans politician finds his idealistic ...  
25    As a volatile young couple who have been toget...  
26    As four couples with different lifestyles go t...  
27    Women who've been sexually brutalized in war-t...  
28    Take an intimate look at the emotionally charg...  
29    When a family moves into an upscale housing de...  
...                                                 ...  
6204  Rachel shocks her proper British parents when ...  
6205  On a tiny island, Pororo the penguin has fun a...  
6206  A child star in the '80s, Samantha clings to t...  
6207  Mixing interviews with dramatic re-enactments,...  
6208  Discovering she's a succubus who sustains hers...  
6209  After Adam graduates from college at age 14, h...  
6210  A Spanish con man masquerades as a therapist a...  
6211  Some have big feet or a sniffly nose, others a...  
6212  In 19th-century Istanbul, a young police offic...  
6213  Destiny brings Mecnun and Leyla together as ne...  
6214  It's not her first talk show, but it is a firs...  
6215  Still pining for Josh, the boy who dumped her ...  
6216  Ms. Frizzle's kid sister Fiona takes the wheel...  
6217  Still rebounding from a breakup, Jessica Day m...  
6218  Full of funny one-liners and always ready for ...  
6219  Ash and his Pikachu travel to the Alola region...  
6220  In Justin's dreams, he and his imaginary frien...  
6221  A new set of six men and women start their liv...  
6222  A suburban mother starts selling marijuana to ...  
6223  On the surface, the Social Welfare Agency appe...  
6224  This CNN original series has chef Anthony Bour...  
6225  Frasier Crane is a snooty but lovable Seattle ...  
6226  This irreverent sitcom featues Ludovico, Feder...  
6227  Imagine your worst fears, then multiply them: ...  
6228  A wacky rabbit and his gang of animal pals hav...  
6229  This parody of first-person shooter games, mil...  
6230  Marc Maron stars as Marc Maron, who interviews...  
6231  Nursery rhymes and original music for children...  
6232  Set during the Russian Revolution, this comic ...  
6233  This hit sitcom follows the merry misadventure...  

[6234 rows x 12 columns]
학습을 수행합니다.

학습 진행률:   0%|          | 0.00/20.0 [00:00<?, ?it/s]
학습 진행률:   5%|▌         | 1.00/20.0 [00:00<00:08, 2.29it/s]
학습 진행률:  10%|█         | 2.00/20.0 [00:00<00:07, 2.28it/s]
학습 진행률:  15%|█▌        | 3.00/20.0 [00:01<00:07, 2.28it/s]
학습 진행률:  20%|██        | 4.00/20.0 [00:01<00:07, 2.28it/s]
학습 진행률:  25%|██▌       | 5.00/20.0 [00:02<00:06, 2.28it/s]
학습 진행률:  30%|███       | 6.00/20.0 [00:02<00:06, 2.27it/s]
학습 진행률:  35%|███▌      | 7.00/20.0 [00:03<00:05, 2.28it/s]
학습 진행률:  40%|████      | 8.00/20.0 [00:03<00:05, 2.28it/s]
학습 진행률:  45%|████▌     | 9.00/20.0 [00:03<00:04, 2.28it/s]
학습 진행률:  50%|█████     | 10.0/20.0 [00:04<00:04, 2.28it/s]
학습 진행률:  55%|█████▌    | 11.0/20.0 [00:04<00:04, 2.21it/s]
학습 진행률:  60%|██████    | 12.0/20.0 [00:05<00:03, 2.21it/s]
학습 진행률:  65%|██████▌   | 13.0/20.0 [00:05<00:03, 2.22it/s]
학습 진행률:  70%|███████   | 14.0/20.0 [00:06<00:02, 2.24it/s]
학습 진행률:  75%|███████▌  | 15.0/20.0 [00:06<00:02, 2.25it/s]
학습 진행률:  80%|████████  | 16.0/20.0 [00:07<00:01, 2.26it/s]
학습 진행률:  85%|████████▌ | 17.0/20.0 [00:07<00:01, 2.26it/s]
학습 진행률:  90%|█████████ | 18.0/20.0 [00:07<00:00, 2.26it/s]
학습 진행률:  95%|█████████▌| 19.0/20.0 [00:08<00:00, 2.26it/s]
학습 진행률: 100%|██████████| 20.0/20.0 [00:08<00:00, 2.27it/s]
                                                          
학습이 완료되었습니다.
```
