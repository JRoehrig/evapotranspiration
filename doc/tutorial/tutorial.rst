Tutorial
========

Package `evapotranspiration` implements the module `penman_monteith_daily`, which calculates daily potential
evapotranspiration (mm/day) according to the Penman-Monteith equation 3 as described in
`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
(Allen et al., 1998).

.. math::

   ET = \frac{\Delta (R_n - G) + \rho_a c_p \frac{e_s - e_a}{r_a}}
   {\lambda \left[ \Delta + \gamma \left( 1 + \frac{r_s}{r_a} \right) \right]}
   \tag{eq. 3, p. 19}

By default the evapotranspiration for a hypothetical grass reference crop is calculated. The required data is:

* latitude, elevation
* date (or day of the year), air temperature, relative air humidity, wind speed, and cloudless hours
  (or shortwave radiation)

Further data is required if reference evapotranspiration is calculated for another crop (e.g., alfalfa).

Installation
____________

::

	pip install evapotranspiration


Quickstart
__________

For given elevation and latitude, different values of ETo can be calculated individually.
Input parameters can be provided in different combinations:

* day of year (doy [-]) or date (date [YYY-mm-dd])
* air temperature: minimum (t_min), maximum (t_max), or mean (t_mean) [°C]
* relative air humidity: minimum (rh_min), maximum (rh_max), or mean (rh_mean) [%]
* number of cloudless hours (n [hours]) or shortwave radiation (rs [MJ/m².day])

Return: reference evapotranspiration [mm/day]

.. code-block::

    >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
    >>> et0_187 = pm.et0(doy=187, u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, n=9.25)
    >>> et0_188 = pm.et0(doy=188, u2=1.553, t_min=14.2, t_max=23.9, rh_min=68, rh_max=78, n=7.3)
    >>> print(et0_187)
    3.872968723753793
    >>> print(et0_188)
    3.7135214054227945

Using `date` and shortwave radiation (rs):

.. code-block::

    >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
    >>> date = '2019-07-06'
    >>> et0 = pm.et0(date=date, u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, rs=22)
    >>> print(et0)
    3.8652694092853253

The following parameters are available after each calculation:

.. code-block::

    >>> print(pm.doy)
    187
    >>> print(pm.u2)
    2.078
    >>> print(pm.lamda)
    2.4610990999999998
    >>> print(pm.s)
    0.12211265844598747
    >>> print(pm.psych)
    0.06658213300847304
    >>> print(pm.mn)
    16.104611680362108
    >>> print(pm.es)
    1.9974855625338357
    >>> print(pm.ea)
    1.4086238018595982
    >>> print(pm.ra)
    41.08837556354228
    >>> print(pm.rs)
    22.072051614368547
    >>> print(pm.rs0)
    30.898458423783794
    >>> print(pm.rns)
    16.995479743063783
    >>> print(pm.rnl)
    3.7117830478654503
    >>> print(pm.rn)
    13.283696695198332
    >>> print(pm.etr)
    2.8051216802858416
    >>> print(pm.etw)
    1.0678470434679512
    >>> print(pm.et)
    3.872968723753793

Using pandas DataFrame:

.. code-block::

    df = pd.DataFrame({'date': ['2001-07-06', '2001-07-06'], 'u2': [2.078, 2.078],
                       't_min': [12.3, 12.3], 't_max': [21.5, 21.5],
                       'rh_min': [63, 63], 'rh_max': [84, 84], 'n': [9.25, 9.25]})

.. code-block::

    pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
    df = pm.et0_frame(df)
    print(df)

::

             date     u2  t_min  t_max  ...       Rnl      ET0r      ET0w       ET0
    0  2001-07-06  2.078   12.3   21.5  ...  3.711783  2.805122  1.067847  3.872969
    1  2001-07-06  2.078   12.3   21.5  ...  3.711783  2.805122  1.067847  3.872969

    [2 rows x 19 columns]