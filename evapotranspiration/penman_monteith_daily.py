import math
import numpy as np
import pandas as pd


class PenmanMonteithDaily(object):
    r"""The class *PenmanMonteithDaily* calculates daily potential evapotranspiration according to the Penman-Monteith
    method as described in
    `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ (Allen et al.,
    1998). Reference evapotranspiration for a hypothetical grass reference crop (:math:`h=12` *cm*;
    :math:`albedo=0.23`, and :math:`LAI=2.88`) is calculated by default. Wind and humidity observations at 2 meters
    height as well as soil heat flux density :math:`G=0.0` *MJ/m²day* are also assumed by default.
    Default values can be changed in the keyword arguments (`**kwargs`) described below.

    The class *PenmanMonteithDaily* solves equation 3 in
    `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_:

    .. math::
       ET = \frac{\Delta (R_n - G) + \rho_a c_p \frac{e_s - e_a}{r_a}}
       {\lambda \left[ \Delta + \gamma \left( 1 + \frac{r_s}{r_a} \right) \right]}
       \tag{eq. 3, p. 19}


    :param elevation: elevation above sea level (*z*) *[m]*. Used in :meth:`clear_sky_shortwave_radiation` and
        :meth:`atmospheric_pressure`
    :type elevation: float
    :param latitude: latitude (:math:`\varphi`) *[decimal degrees]*. Used in :meth:`sunset_hour_angle` and
        :meth:`extraterrestrial_radiation`
    :type latitude: float

    :Keyword Arguments:

       * **albedo** (*float*) - albedo or canopy reflection coefficient (:math:`\alpha`) *[-]*.
         Range: :math:`0.0  \leq \alpha \leq 1.0`. Default :math:`albedo=0.23` for the hypothetical grass
         reference crop. Used in :meth:`net_shortwave_radiation`
       * **h** (*float*) - crop height (*h*) *[m]*. Default :math:`h=0.12` for the hypothetical grass reference
         crop. Required to calculate the zero plane displacement height (:math:`d`) *[m]* and the roughness length
         governing momentum (:math:`z_{om}`) *[m]*, both necessary for the aerodynamic resistance (:math:`r_a`) *[s/m]*.
         See :meth:`aerodynamic_resistance_factor`
       * **lai** (*float*) - leaf area index (:math:`LAI`) *[-]*. Default :math:`lai=2.88` for the hypothetical
         grass reference crop. See *BOX 5* in
         `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ and
         :meth:`bulk_surface_resistance`
       * **rl** (*float*) - bulk stomatal resistance of well-illuminated leaf (:math:`r_l`) *[s/m]*. Default
         :math:`rl=100.0` for any crop. See :meth:`bulk_surface_resistance`
       * **zm** (*float*) - height of wind measurements (:math:`z_m`) *[m]*. Default :math:`zm=2.0`. Required to
         calculate aerodynamic resistance (:math:`r_a`) *[s/m]*. See :meth:`aerodynamic_resistance_factor`
       * **zh** (*float*) - height of humidity measurements (:math:`z_h`) *[m]*. Default :math:`zh=2.0`. Required to
         calculate aerodynamic resistance (:math:`r_a`) *[s/m]*. See :meth:`aerodynamic_resistance_factor`
       * **g** (*float*) - soil heat flux density (:math:`G`) *[MJ/m²day]*. Default :math:`g=0.0`. This
         corresponds to :math:`G` in eq. 3, p. 19 above. It can be also given with daily parameters in :meth:`et0`

    .. note::
        Only :attr:`elevation` and :attr:`latitude` are mandatory parameters of :meth:`PenmanMonteithDaily()`.
        :attr:`albedo`, :attr:`h`, and :attr:`lai` are only necessary when calculating evapotranspiration for crops
        other than reference grass.

    :ivar doy: day of year *[-]*
    :ivar z: elevation in meters above sea level (*z*) *[m]*
    :ivar p: atmospheric pressure (*P*) *[kPa]*
    :ivar u2: wind speed at height :math:`z` (:math:`u_2`) *[m/s]*
    :ivar ld: latent heat of vaporization (:math:`\lambda`) *[MJ/kg]*. See :meth:`latent_heat_of_vaporization()`
    :ivar s: slope of saturation vapour pressure curve (:math:`\Delta`) *[kPa/°C]*.
        See :meth:`slope_of_saturation_vapour_pressure_curve()`
    :ivar psych: psychrometric constant (:math:`\gamma`) *[kPa/°C]*. See :meth:`psychrometric_constant()`
    :ivar mn: daylight hours (:math:`N`) *[hours]*. See :meth:`daylight_hours()`
    :ivar es: saturation vapour pressure (:math:`e_s`) *[kPa]*. See :meth:`saturation_vapour_pressure()`
    :ivar ea: actual vapour pressure (:math:`e_a`) *[kPa]*. See :meth:`actual_vapour_pressure()`
    :ivar ra: daily extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*.  See :meth:`extraterrestrial_radiation()`
    :ivar rs: daily shortwave radiation (:math:`R_s`) *[MJ/m²day]*. See :meth:`shortwave_radiation()`
    :ivar rs0: clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*.
        See :meth:`clear_sky_shortwave_radiation()`
    :ivar rns: net shortwave radiation (:math:`R_{ns}`) *[MJ/m²day]*. See :meth:`net_shortwave_radiation()`
    :ivar rnl: net outgoing longwave radiation (:math:`R_{nl}`) *[MJ/m²day]*. See :meth:`net_longwave_radiation()`
    :ivar rn: net radiation (:math:`R_{n}`) *[MJ/m²day]*. :math:`R_{n} = R_{ns} - R_{nl}`
    :ivar etr: radiation component of reference evapotranspiration *[mm/day]*
    :ivar etw: wind component of reference evapotranspiration *[mm/day]*
    :ivar et: reference evapotranspiration *[mm/day]*

    Object Constants:
        * **e** - ratio molecular weight of water vapour/dry air (:math:`\varepsilon`) *[-]*.
          :math:`e = 0.622`
        * **r** - specific gas constant *[kJ/kg.K]*. :math:`r = 0.287`
        * **k** - von Karman constant (:math:`k`) *[-]*, see
          `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ eq. 4.
          :math:`k=0.41`

    Object crop specific factors:
        * **d_factor** - factor of the zero plane displacement height (:math:`d`) *[-]*. :math:`d\_factor = 2.0 / 3.0`
        * **zom_factor** - factor of the roughness length governing momentum transfer (:math:`z_{om}`) *[-]*.
          :math:`zom\_factor = 0.123`
        * **zoh_factor** - factor of the roughness length governing transfer of heat and vapour (:math:`z_{oh}`) *[-]*.
          :math:`zoh\_factor = 0.1`
        * **lai_active_factor** - factor of the active (sunlit) leaf area index (:math:`LAI_{active}`) *[-]* (it
          considers that generally only the upper half of dense clipped grass is actively contributing to the surface
          heat and vapour transfer). :math:`lai\_active\_factor = 0.5`

    Calculation with :meth:`et0`::

        - pm = PenmanMonteithDaily(elevation, latitude, ...)
        - et0 = pm.et0(...)

    Calculation with :meth:`et0_frame` given a *pandas.DataFrame()* as input parameter::

        - pm = PenmanMonteithDaily(elevation, latitude, ...)
        - df = pm.et0_frame(df, ...)

    """

    def __init__(self, elevation, latitude, **kwargs):
        self.albedo = kwargs.get('albedo', 0.23)  # albedo
        self.h = kwargs.get('h', 0.12)            # crop height h [m]
        self.zm = kwargs.get('zm', 2.0)           # height of wind measurements [m]
        self.zh = kwargs.get('zh', 2.0)           # roughness length governing transfer of heat and vapour [m]
        self.lai = kwargs.get('lai', 2.88)        # LAI dependence
        self.rl = kwargs.get('rl', 100.0)         # The stomatal resistance
        self.g_default = kwargs.get('g', 0.0)     # soil heat flux density [MJ/m²day]

        self.doy = None
        self.u2 = None
        self.ld = None
        self.s = None
        self.pc = None
        self.mn = None
        self.es = None
        self.ea = None
        self.ra = None
        self.rs = None
        self.rs0 = None
        self.rns = None
        self.rnl = None
        self.rn = None
        self.etr = None
        self.etw = None
        self.et = None

        self.e = 0.622
        self.r = 0.287
        self.k = 0.41
        self.d_factor = 2.0 / 3.0
        self.zom_factor = 0.123
        self.zoh_factor = 0.1
        self.lai_active_factor = 0.5

        if latitude:
            days = np.array(range(367))
            latitude = float(np.radians(latitude))
            dr_366 = self.inverse_relative_distance_earth_sun(days)
            sd_366 = np.array([self.solar_declination(day) for day in range(367)])
            ws_366 = np.array([self.sunset_hour_angle(latitude, s) for s in sd_366])
            self.daylight_hours_366 = np.array([PenmanMonteithDaily.daylight_hours(w) for w in ws_366])
            self.ra_366 = np.array([self.extraterrestrial_radiation(
                dr_366[i], ws_366[i], latitude, sd_366[i]) for i in range(len(dr_366))])
            self.rs0_366 = np.array([self.clear_sky_shortwave_radiation(
                ra, elevation=elevation) for ra in self.ra_366])
        else:
            self.daylight_hours_366 = None
            self.ra_366 = None
            self.rs0_366 = None

        self.z = elevation

        self.p = PenmanMonteithDaily.atmospheric_pressure(self.z)

        ra_factor = self.aerodynamic_resistance_factor()

        self.f1 = 86400 * self.e / (1.01 * self.r * ra_factor)
        """f1 = (specific heat at constant pressure) * (mean air density at constant pressure) /
             (1.01 * :attr:`r` * :meth:`aerodynamic_resistance_factor`). 
             `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ Box 6
        """

        self.f2 = self.bulk_surface_resistance() / ra_factor
        r""":math:`f_1 = \frac{rs}{f_{ra}}` with :math:`f_{ra}` = :meth:`aerodynamic_resistance_factor`"""

    def reset(self):
        r"""Reset the following output attributes before calculating :math:`ETo`: :math:`doy`, :math:`u2`,
            :math:`ld`, :math:`s`, :math:`pc`, :math:`mn`, :math:`es`, :math:`ea`, :math:`ra`,
            :math:`rs`, :math:`rs0`, :math:`rns`, :math:`rnl`, :math:`rn`, :math:`etr`, :math:`etw`, and :math:`et`
        """
        self.doy = None
        self.u2 = None
        self.ld = None
        self.s = None
        self.pc = None
        self.mn = None
        self.es = None
        self.ea = None
        self.ra = None
        self.rs = None
        self.rs0 = None
        self.rns = None
        self.rnl = None
        self.rn = None
        self.etr = None
        self.etw = None
        self.et = None

    @staticmethod
    def atmospheric_pressure(z):
        r""" Return the atmospheric pressure (:math:`P`) *[kPa]* as a function of the elevation above sea level as
        defined in `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 7, p. 31):

        .. math::

            P = 101.3\left(\frac{293-0.0065z}{293}\right)^{5.26}

        The atmospheric pressure (:math:`P`) is the pressure exerted by the weight of the earth's atmosphere.
        Evaporation at high altitudes is promoted due to low atmospheric pressure as expressed in the psychrometric
        constant. The effect is, however, small and in the calculation procedures, the average value for a location
        is sufficient. A simplification of the ideal gas law, assuming :math:`20` *°C* for a standard atmosphere,
        can be employed to calculate :math:`P`
        (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_).

        :param z: elevation above sea level *[m]*
        :type z: float or np.array
        :return: (*float or np.array*) atmospheric pressure (:math:`P`) *[kPa]*
        """
        return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26

    @staticmethod
    def latent_heat_of_vaporization(temperature=20):
        r"""Return the latent heat of vaporization (:math:`\lambda`) *[MJ/kg]* as described in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (Annex 3, eq. 3-1, p. 223):

        .. math::

            \lambda = 2.501-(2.361 * 10^{-3})T

        :param temperature: air temperature (:math:`T`) *[°C]*. Default :math:`temperature=20`
        :type temperature: float or np.array
        :return: (*float or np.array*) latent heat of vaporization (:math:`\lambda`) *[MJ/kg]*.
            Default :math:`\lambda=2.45378`
        """
        return 2.501 - 2.361e-3 * temperature

    @staticmethod
    def psychrometric_constant(p, **kwargs):
        r"""Return the psychrometric constant (:math:`\gamma`) *[kPa/°C]* according to
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        eq. 8, p. 32:

        .. math::

            \gamma = \frac{c_p P}{\varepsilon \lambda}

        or, using default values:

        .. math::

            \gamma = a_{psy} \cdot P

        :param p: atmospheric pressure (:math:`P`) *[kPa]*
        :type p: float or np.array

        :Keyword Arguments:

           * **lamda** (*float*) - latent heat of vaporization (:math:`\lambda`) *[MJ/kg]*. Default :math:`lamda=2.45`.
             See Used in :meth:`latent_heat_of_vaporization`
           * **cp** (*float*) - specific heat at constant pressure (:math:`c_p`) *[MJ/kg]*. Default
             :math:`cp=1.013e^{-3}`
           * **epsilon** (*float*) - ratio molecular weight of water vapour/dry air (:math:`\epsilon`) *[-]*.
             Default :math:`epsilon=0.622`
           * **a_psy** (*float*) - coefficient depending on the type of the ventilation of the bulb *[1/°C]*. Examples:

             * :math:`a_{psy} = 0.000665` (default)
             * :math:`a_{psy} = 0.000662` for ventilated (Asmann type) psychrometers, with an air movement of some 5
               *m/s*
             * :math:`a_{psy} = 0.000800` for natural ventilated psychrometers (about 1 *m/s*)
             * :math:`a_{psy} = 0.001200` for non-ventilated psychrometers installed indoors

        The method uses :math:`a_{psy}` if given, otherwise eq. 8 (see above) with given or default values. Default
        values correspond to :math:`a_{psy} = 0.000665` as argument.

        :return: (*float or np.array*) psychrometric constant (:math:`\gamma`) *[kPa/°C]*
        """
        if 'a_psy' in kwargs:
            return kwargs.get('a_psy', 0.000665) * p
        else:
            return (kwargs.get('cp', 1.013e-3) * p) / (kwargs.get('epsilon', 0.622) * kwargs.get('lamda', 2.45))

    @staticmethod
    def saturation_vapour_pressure(*temperature):
        r"""Return the saturation vapour pressure (:math:`e_s`) *[kPa]* according to
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 11, p. 36):

        .. math::

             e^{°}(T) = 0.6108 exp \left[\frac{17.27 T}{T + 237.3}\right]

        :param temperature: air temperature (:math:`T`) *[°C]*
        :type temperature: float or np.array
        :return: (*float or np.array*) saturation vapour pressure (:math:`e_s`) *[kPa]*
        """
        t = np.array([0.6108 * np.exp((17.27 * t) / (t + 237.3)) for t in temperature])
        t = np.mean(t, axis=0)
        return t

    @staticmethod
    def slope_of_saturation_vapour_pressure_curve(*temperature):
        r"""Return the slope of saturation vapour pressure curve (:math:`\Delta`) *[kPa/°C]* according to
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 13, p. 37):

         .. math::

             \Delta = 4098\left[\frac{0.6108exp\left(\frac{17.27 T}{T + 237.3}\right)}{(T + 237.3)^{2}}\right]

        :param temperature: air temperature (:math:`T`) *[°C]*
        :type temperature: float or np.array
        :return: (*float or np.array*) slope of saturation vapour pressure curve (:math:`\Delta`) *[kPa/°C]*
        """
        sl = np.array([(4098.0 * PenmanMonteithDaily.saturation_vapour_pressure(t)) / ((t + 237.3) ** 2)
                       for t in temperature])
        return np.mean(sl, axis=0)

    @staticmethod
    def actual_vapour_pressure(**kwargs):
        """Return the actual vapour pressure (:math:`e_a`) *[kPa]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (p. 37 , 38 , and 39):

        :Keyword Arguments:

           * **rh_min** (*float*) - 0.0 to 100.0 *[%]*
           * **rh_max** (*float*) - 0.0 to 100.0 *[%]*
           * **es_min** (*float*) - saturation vapour pressure for :math:`t\_min` *[kPa]*
           * **es_max** (*float*) - saturation vapour pressure for :math:`t\_max` *[kPa]*
           * **t_min** (*float*) - minimum air temperature *[°C]*
           * **t_max** (*float*) - maximum air temperature *[°C]*
           * **t_dew** (*float*) - dew point temperature *[°C]*
           * **t_wet** (*float*) - wet bulb temperature *[°C]*
           * **t_dry** (*float*) - dry bulb temperature *[°C]*
           * **apsy** (*float*) - coefficient depending on the type of ventilation of the wet bulb *[-]*

        :return: (*float or np.array*) actual vapour pressure (:math:`e_a`) *[kPa]*
        """
        try:
            rh_min = kwargs['rh_min'] / 100.0
            rh_max = kwargs['rh_max'] / 100.0
            if 'es_min' in kwargs and 'es_max' in kwargs:
                es_min = kwargs['es_min']
                es_max = kwargs['es_max']
            else:
                es_min = PenmanMonteithDaily.saturation_vapour_pressure(kwargs['t_min'])
                es_max = PenmanMonteithDaily.saturation_vapour_pressure(kwargs['t_max'])
            return (rh_max * es_min + rh_min * es_max) / 2.0
        except KeyError:
            t_dew = kwargs.get('t_dew', None)
            return 0.6108 * math.exp((17.27 * t_dew) / (t_dew + 237.3))

    def aerodynamic_resistance_factor(self):
        r"""Return the aerodynamic resistance (:math:`r_a`) *[s/m]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 4, p. 20):

        .. math::

            r_a = \frac{ \ln \left( \frac{z_m - d}{z_{om}} \right) \ln \left( \frac{z_h - d}{z_{oh}} \right) }
            { k^2 u_z }

        where (see :meth:`PenmanMonteithDaily()`):

            :math:`u_z` --- the wind speed *[m/s]* at height :math:`z` (see :meth:`et0()`)

            :math:`k` --- von Karman's constant *[-]*

            :math:`zm` --- height of wind measurements *[m]*

            :math:`zh` --- height of air humidity measurements *[m]*

        The aerodynamic resistance factor :math:`f_{r_a}` is constant for a given crop:

        .. math::

            f_{r_a} = \frac{ \ln \left( \frac{z_m - d}{z_{om}} \right) \ln \left( \frac{z_h - d}{z_{oh}} \right) }
            { k^2}

        with the zero plane displacement height (:math:`d`):

        .. math::

            d = f_d  \cdot h

        and roughness length governing momentum transfer (:math:`z_{om}`):

        .. math::

            z_{om} = f_{zom}  \cdot  h

        where:

            :math:`f_d` --- defined in :attr:`d_factor`

            :math:`f_{zom}` --- defined in in :attr:`zom_factor`

        :return: (*float*) aerodynamic resistance factor :math:`f_{r_a}`
        """

        # zero plane displacement height, d [m]
        d = self.d_factor * self.h

        # roughness length governing momentum transfer [m]
        zom = self.zom_factor * self.h

        # roughness length governing transfer of heat and vapour [m]
        zoh = self.zoh_factor * zom

        return math.log((self.zm - d) / zom) * math.log((self.zh - d) / zoh) / (self.k ** 2)

    def bulk_surface_resistance(self):
        r"""Return (bulk) surface resistance (:math:`r_s`) *[s/m]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 5, p. 21):

        .. math::

            r_s = \frac{ r_l } { LAI_{active} }

        where:

            :math:`r_l` --- the bulk stomatal resistance of the well-illuminated leaf *[s/m]*

            :math:`LAI_{active}` --- the active (sunlit) leaf area index *[m² (leaf area) / m² (soil surface)]*

        A general equation for :math:`LAI_{active}` is:

        .. math::

            LAI_{active} = 0.5 LAI

        with:

        .. math::

            LAI = 24 h

        where :math:`h` is an optional input parameter in :class:`PenmanMonteithDaily`.

        :return: (*float*) (bulk) surface resistance :math:`r_s` *[s/m]*
        """
        #
        # active (sunlit) leaf area index [m^2 (leaf area) / m^2 (soil surface)]
        lai_active = self.lai_active_factor * self.lai

        rs = self.rl / lai_active
        return rs

    @staticmethod
    def to_u2(uz, z):
        r""" Return the calculated wind speed at 2 meters above ground surface (:math:`u_2`) *[m/s]*:

        .. math::

            u_2 = \frac{ 4.87 u_z}{ \ln{(67.8 z - 5.42)}}

        :param uz: measured wind speed at :math:`z` meters above ground surface *[m/s]*
        :type uz: float or np.array
        :param z: height of measurement above ground surface *[m]*
        :type z: float
        :return: (*float or np.array*) wind speed at 2 meters above ground surface *[m/s]*
        """
        return uz * 4.87 / np.log(67.8 * z - 5.42)

    @staticmethod
    def extraterrestrial_radiation(dr, ws, lat, sd):
        r"""Return the extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 21, p. 46):

        .. math::

            R_a = \frac{24(60)}{\pi} G_{sc} d_r [ \omega_s \sin(\varphi) \sin(\delta) + \cos(\varphi) \cos(\delta)
            \sin(\omega_s)]

        :param dr: inverse relative distance Earth-Sun (:math:`d_r`) *[-]*.
            See :meth:`inverse_relative_distance_earth_sun`
        :type dr: float
        :param ws: sunset hour angle (:math:`\omega_s`) *[rad]*. See :meth:`sunset_hour_angle`
        :type ws: float
        :param lat: latitude (:math:`\varphi`) *[rad]*
        :type lat: float
        :param sd: solar declination (:math:`\delta`) *[rad]*. See :meth:`solar_declination`
        :type sd: float
        :return: *(float or np.array)* daily extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*
        """
        # solar_constant = 0.0820 # MJ.m-2.min-1
        # (24.0 * 60.0 / pi) * solar_constant = 37.586031360582005
        return 37.586031360582005 * dr * (ws * np.sin(lat) * np.sin(sd) + np.cos(lat) * np.cos(sd) * np.sin(ws))

    @staticmethod
    def inverse_relative_distance_earth_sun(day):
        r"""Return the inverse relative distance Earth-Sun (:math:`d_r`) *[-]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 23, p. 46):

        .. math::

            d_r = 1 + 0.033 \cos{ \left( \frac{2 \pi}{365} J \right)}

        :param day: day of the year (:math:`J`) *[-]*. Range: :math:`1 \leq J \leq 366`
        :type day: int or np.array
        :return: *(float or np.array)* inverse relative distance Earth-Sun (:math:`d_r`) *[-]*
        """
        # 2.0 * pi / 365 = 0.01721420632103996
        return 1 + 0.033 * np.cos(0.01721420632103996 * day)

    @staticmethod
    def solar_declination(day):
        r"""Return the solar declination (:math:`\delta`) *[rad]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 24, p. 46):

        .. math::

            \delta = 0.409 \sin{ \left( \frac{2 \pi}{365} J - 1.39\right)}

        :param day: day of the year (:math:`J`) *[-]*. Range: :math:`1 \leq J \leq 366`
        :type day: int
        :return: (*float or np.array*) solar declination (:math:`\delta`) *[rad]*
        """
        # 2.0 * pi / 365 = 0.01721420632103996
        return 0.409 * np.sin(0.01721420632103996 * day - 1.39)

    @staticmethod
    def sunset_hour_angle(lat, sd):
        r"""Return the sunset hour angle (:math:`\omega_s`) *[rad]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 25, p. 46):

        .. math::

            \omega_s = \arccos{ \left[-tan(\varphi)tan(\delta)\right]}

        :param lat: latitude (:math:`\varphi`) *[rad]*
        :type lat: float or np.array
        :param sd: solar declination (:math:`\delta`) *[rad]*. See :meth:`solar_declination`
        :type sd: float or np.array
        :return: (*float or np.array*) sunset hour angle (:math:`\omega_s`) *[rad]*
        """
        return np.arccos(-np.tan(sd) * np.tan(lat))

    @staticmethod
    def daylight_hours(ws):
        r"""Return the daylight hours (:math:`N`) *[hour]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 34, p. 49):

        .. math::

            N = \frac{24}{\pi} \omega_s

        :param ws: sunset hour angle (:math:`\omega_s`) *[rad]*. See :meth:`sunset_hour_angle`
        :type ws: float or np.numpy
        :return: (*float or np.numpy*) daylight hours (:math:`N`) *[hour]*
        """
        # 24.0 / pi = 7.639437268410976
        return 7.639437268410976 * ws

    @staticmethod
    def clear_sky_shortwave_radiation(ra, elevation=0.0, a_s=0.25, b_s=0.50):
        r"""Return the clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*. It is required for computing
        :meth:`net_longwave_radiation`.

        For near sea level or when calibrated values for :math:`a_s` and :math:`b_s` are available
        (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 36,
        p. 51):

        .. math::

           R_{so} = (a_s + b_s ) R_a

        When calibrated values for :math:`a_s` and :math:`b_s` are not available
        (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_,
        eq. 37, p. 51):

        .. math::

           R_{so} = (0.75 + 2 * 10^{−5} z) R_a

        where :math:`z` is the station elevation above sea level *[m]*.

        :param ra: extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*. See :meth:`extraterrestrial_radiation`
        :type ra: float or np.numpy
        :param elevation: meters above sea level see (:math:`z`) [m]. See :attr:`elevation`
        :type elevation: float or np.numpy
        :param a_s: regression constant (:math:`a_s`) *[-]*. Default :math:`a_s=0.25`. It expresses the fraction of
            extraterrestrial radiation reaching the earth on overcast days (:math:`n = 0`)
        :type a_s: float or np.numpy
        :param b_s: regression constant (:math:`b_s`) *[-]*. Default :math:`b_s=0.50`. The expression
            :math:`a_s+b_s` indicates the fraction of extraterrestrial radiation reaching the earth on clear days
            (:math:`n = N`)
        :type b_s: float or np.numpy
        :return: (*float or np.numpy*) daily clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*
        """
        rs0 = ((a_s + b_s) + 2e-5 * elevation) * ra
        return rs0

    @staticmethod
    def shortwave_radiation(ra, n, mn, a_s=0.25, b_s=0.50):
        r"""Return the daily shortwave radiation (:math:`R_s`) *[MJ/m²day]* according to the Angstrom formula as
        described in `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 35, p. 50):

        .. math::

           R_s = \left( a_s + b_s \frac{n}{N} \right) R_a

        Depending on atmospheric conditions (humidity, dust) and solar declination (latitude and month), the Angstrom
        values :math:`a_s` and :math:`b_s` will vary. Where no actual solar radiation data are available and no
        calibration has been carried out for improved :math:`a_s` and :math:`b_s` parameters, the values
        :math:`a_s = 0.25` and :math:`b_s = 0.50` are recommended.

        :param ra: extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*. See :meth:`extraterrestrial_radiation`
        :type ra: float or np.array
        :param n: actual duration of sunshine or cloudless hours (:math:`n`) *[hour]*
        :type n: float or np.array
        :param mn: maximum possible duration of sunshine or daylight hours (:math:`N`) *[hour]*
            See :meth:`daylight_hours`
        :type mn: float, np.array
        :param a_s: regression constant (:math:`as`) *[-]*. Default :math:`a_s=0.25`. It expresses the fraction
            of extraterrestrial radiation reaching the earth on overcast days (:math:`n = 0`)
        :type a_s: float or np.numpy
        :param b_s: regression constant (:math:`bs`) *[-]*. Default :math:`b_s=0.50`. The expression
            :math:`a_s+b_s` indicates the fraction of extraterrestrial radiation reaching the earth on clear days
            (:math:`n = N`)
        :type b_s: float or np.numpy
        :return: (*float, np.array*) daily total shortwave radiation (:math:`R_s`) *[MJ/m²day]* reaching the earth

        .. note::
                If shortwave radiation (i.e., solar radiation) measurements are available, :meth:`shortwave_radiation`
                function is no needed. Measurements of shortwave radiation may be directly used as input data in
                :meth:`et0`.

        """
        rns = (a_s + b_s * n / mn) * ra
        return rns

    @staticmethod
    def net_shortwave_radiation(rs, albedo):
        r"""The net shortwave radiation (:math:`R_{ns}`) *[MJ/m²day]* resulting from the balance between incoming
        and reflected solar radiation as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 38, p. 51):

        .. math::

            R_{ns} = (1 − \alpha) R_s

        :param rs: daily shortwave radiation (:math:`R_s`) *[MJ/m²day]*. See :meth:`shortwave_radiation`
        :type rs: float or np.array
        :param albedo: albedo or reflection coefficient (:math:`\alpha` *[-]*). Range:
            :math:`0.0  \leq \alpha \leq 1.0` (:math:`\alpha=0.23` for the hypothetical grass reference crop).
            See :class:`PenmanMonteithDaily` and :meth:`et0`
        :type albedo: float or np.array
        :return: (*float or np.array*) daily net shortwave radiation (:math:`R_{ns}`) *[MJ/m²day]* reaching the earth
        """
        return (1.0 - albedo) * rs

    @staticmethod
    def net_longwave_radiation(t_min, t_max, rs, rs0, ea=None):
        r"""Return the net outgoing longwave radiation (:math:`R_{nl}`) *[MJ/m²day]* as defined in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_
        (eq. 39, p. 52):

        .. math::

            R_{nl} = \sigma\left[\frac{T_{max,K}^4 + T_{min,K}^4}{2}\right](0.34-0.14\sqrt{e_a})\left(1.35
            \frac{R_s}{R_{so}}-0.35\right)

        :param t_min: minimum daily air temperature (:math:`T_{max}`) *[°C]*
        :type t_min: float or np.array
        :param t_max: maximum daily air temperature (:math:`T_{min}`) *[°C]*
        :type t_max: float or np.array
        :param rs: shortwave radiation (:math:`R_s`) *[MJ/m²day]*. See :meth:`shortwave_radiation`
        :type rs: float or np.array
        :param rs0: clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*. See
            :meth:`clear_sky_shortwave_radiation`
        :type rs0: float or np.array
        :param ea: actual vapour pressure (:math:`e_a`) *[kPa]*
        :type ea: float or np.array
        :return: (*float or np.array*) daily net outgoing longwave radiation (:math:`R_{nl}`) *[MJ/m²day]*

        .. note::
                The :math:`R_s/R_{so}` term in the equation above must be limited so that :math:`R_s/R_{so} \leq 1.0`.

        """
        t_min = t_min + 273.15
        t_max = t_max + 273.15
        if ea is not None:
            rln = 4.903e-9 * (t_min ** 4 + t_max ** 4) * 0.5 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * rs / rs0 - 0.35)
        else:
            t_mean = (t_min + t_max) / 2.0
            rln = 4.903e-9 * (t_min ** 4 + t_max ** 4) * 0.5 * \
                (-0.02 + 0.261 * np.exp(-7.77e10 ** -4 * t_mean ** 2)) * (1.35 * rs / rs0 - 0.35)
        return rln

    def et0(self, **kwargs):
        r"""Returns potential evapotranspiration (:math:`ETo`) *[mm/day]* as described in
        `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_. Reference
        (grass) potencial evapotranspiration is returned for default constructor values. If values in `**kwargs` are
        arrays, their lengths must be the same.

        :Keyword Arguments:

           * **date** (*str, datetime.date, datetime.datetime, pandas.TimeStamp, or np.array*)
           * **doy** (*int or np.array*) - day of the year (:math:`J`) *[-]*. Range: :math:`1 \leq J \leq 366`.
             It is not used if date is given
           * **u2** (*float or np.array*) - wind speed at 2 meters above ground surface *[m/s]*
           * **uz** (*float or np.array*) - measured wind speed at :math:`z` meters above ground surface *[m/s]*
           * **z** (*float or np.array*) - height of measurement above ground surface *[m]*
           * **t_mean** (*float or np.array*) - daily mean air temperature *[°C]*
           * **t_min** (*float or np.array*) - daily minimum air temperature *[°C]*
           * **t_max** (*float or np.array*) - daily maximum air temperature *[°C]*
           * **rh_mean** (*float or np.array*) - daily mean relative humidity *[%]*
           * **rh_min** (*float or np.array*) - daily minimum relative humidity *[%]*
           * **rh_max** (*float or np.array*) - daily maximum relative humidity *[%]*
           * **rs** (*float or np.array*) - solar or shortwave radiation *[MJ/m²day]*
           * **n** (*float or np.array*) - daily actual duration of sunshine or cloudless hours *[hour]*
           * **g** (*float or np.array*) - soil heat flux density *[MJ/m²day]*. If not given, *g* defined in
             :meth:`PenmanMonteithDaily` will be used
           * **a_s** (*float or np.array*) - see :meth:`shortwave_radiation`. Default :math:`a_s = 0.25`
           * **b_s** (*float or np.array*) - see :meth:`shortwave_radiation`. Default :math:`b_s = 0.50`
           * **negative_rnl** (*bool*) - allow negative net longwave radiation. Default :math:`negative\_rnl=True`
           * **negative_et0** (*bool*) - allow negative reference evapotranspiration. Default :math:`negative\_et0=True`

        :return: (*float or np.array*) potential evapotranspiration (:math:`ETo`) *[mm/day]*

        Cases:

        * If date and doy are given, :math:`doy` is disregarded
        * if :math:`uz` is given, :math:`z` must also be given
        * if :math:`u2` and (:math:`uz`, :math:`z`) are given, both :math:`uz` and :math:`z` are disregarded
        * if :math:`rs` and :math:`n` are given, :math:`n` will be disregarded
        * The best options for air temperature are, in this order: 1) t_min, t_max, and t_mean, 2) t_min, t_max, and
          3) tmean
        * The best options for relative air humidity are, in this order: 1) rh_max and rh_min, 2) rh_max, and 3)
          rh_mean

        Example 1::

            >>> from evapotranspiration.penman_monteith_daily import PenmanMonteithDaily
            >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
            >>> et0 = pm.et0(doy=187, u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, n=9.25)
            >>> print(et0)
            3.872968723753793

        Example 2::

            >>> from evapotranspiration.penman_monteith_daily import PenmanMonteithDaily
            >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
            >>> et0 = pm.et0(date='2001-07-06', u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, n=9.25)
            >>> print(et0)
            3.872968723753793

        Example 3::

            >>> from evapotranspiration.penman_monteith_daily import PenmanMonteithDaily
            >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
            >>> date=np.array(['2001-07-06', '2001-07-06'])
            >>> u2=np.array([2.078, 2.078])
            >>> t_min=np.array([12.3, 12.3])
            >>> t_max=np.array([21.5, 21.5])
            >>> rh_min=np.array([63, 63])
            >>> rh_max=np.array([84, 84])
            >>> n=np.array([9.25, 9.25])
            >>> et0 = pm.et0(date=date, u2=u2, t_min=t_min, t_max=t_max, rh_min=rh_min, rh_max=rh_max, n=n)
            >>> print(et0)
            [3.87296872 3.87296872]
        """

        self.reset()

        try:
            self.u2 = kwargs.get('u2', None)
            if self.u2 is None:
                self.u2 = self.to_u2(kwargs['uz'], kwargs['z'])
        except KeyError:
            raise KeyError('Penmam-Monteith: Either u2 or both uz and z must be given')

        t_min = kwargs.get('t_min', None)
        if t_min is None:
            t_min = kwargs['t_mean']
        t_max = kwargs.get('t_max', None)
        if t_max is None:
            t_max = kwargs['t_mean']
        t_mean = kwargs.get('t_mean', None)

        rh_min = kwargs.get('rh_min', None)
        rh_max = kwargs.get('rh_max', None)
        if rh_max is not None:
            if rh_min is None:
                rh_min = rh_max
        else:
            rh_min = rh_max = kwargs['rh_mean']

        self.doy = kwargs.get('doy', None)
        if self.doy is None:
            self.doy = pd.to_datetime(kwargs['date']).dayofyear

        self.rs = kwargs.get('rs', None)

        n = kwargs.get('n', None)

        g = kwargs.get('g', None)
        if g is None:
            g = self.g_default

        a_s = kwargs.get('a_s', 0.25)
        b_s = kwargs.get('b_s', 0.50)

        if t_mean is None:
            t_mean = (t_min + t_max) / 2.0

        self.ld = PenmanMonteithDaily.latent_heat_of_vaporization(t_mean)
        # In FAO 56, where delta occurs in the numerator and denominator, the slope
        # of the vapour pressure curve is calculated using mean air temperature (Equation 9)
        self.s = PenmanMonteithDaily.slope_of_saturation_vapour_pressure_curve(t_mean)
        self.pc = PenmanMonteithDaily.psychrometric_constant(self.p, lamda=self.ld)

        self.es = PenmanMonteithDaily.saturation_vapour_pressure(t_min, t_max)
        self.ea = PenmanMonteithDaily.actual_vapour_pressure(rh_min=rh_min, rh_max=rh_max, t_min=t_min, t_max=t_max)

        try:
            self.ra = np.array([self.ra_366[i] for i in self.doy])
            self.rs0 = np.array([self.rs0_366[i] for i in self.doy])
            if self.rs is None:
                self.mn = np.array([self.daylight_hours_366[i] for i in self.doy])
                self.rs = self.shortwave_radiation(self.ra, n, self.mn, a_s, b_s)
                # FAO56 eq. 39. The Rs/Rso term in equation 39 must be limited so that Rs/Rso ≤ 1.0.
                self.rs = np.where(self.rs > self.rs0, self.rs0, self.rs)
        except TypeError:
            self.ra = self.ra_366[self.doy]
            self.rs0 = self.rs0_366[self.doy]
            if self.rs is None:
                self.mn = self.daylight_hours_366[self.doy]
                self.rs = self.shortwave_radiation(self.ra, n, self.mn, a_s, b_s)
                # FAO56 eq. 39. The Rs/Rso term in equation 39 must be limited so that Rs/Rso ≤ 1.0.
                self.rs = self.rs0 if self.rs > self.rs0 else self.rs

        self.rns = self.net_shortwave_radiation(self.rs, self.albedo)
        self.rnl = self.net_longwave_radiation(t_min, t_max, self.rs, self.rs0, self.ea)
        if kwargs.get('negative_rnl', False) and self.rnl < 0.0:
            self.rnl = 0.0

        self.rn = self.rns - self.rnl

        # denominator of FAO 56 eq. 3
        etd = self.ld * (self.s + self.pc * (1 + self.f2 * self.u2))

        # ETo energy component of FAO 56 eq. 3
        self.etr = self.s * (self.rn - g) / etd
        # ETo wind component of FAO 56 eq. 3
        self.etw = (self.ld * self.pc * self.u2 * self.f1 * (self.es - self.ea) / (t_mean + 273.0)) / etd
        # Reference evapotranspiration
        self.et = self.etr + self.etw
        self.et = np.where(self.et < 0.0, 0.0, self.et)
        try:
            self.et = float(self.et)
        except TypeError:
            pass
        if kwargs.get('negative_rnl', False) and self.et < 0.0:
            self.et = 0.0
        return self.et

    def et0_frame(self, df, **kwargs):
        """Return the input DataFrame extended by :meth:`et0` and further calculation parameters.

        :param df: pandas DataFrame with columns corresponding to the inputs described in :meth:`et0`
        :type df: pandas.DataFrame

        :Keyword Arguments:

           * **show_all** (*bool*) - show all results if :math:`True`, otherwise set `parameter=True` to show individual
             parameters. For example :math:`doy=True`, :math:`ld=True`, etc. See :meth:`PenmanMonteithDaily`

        :return: (*pandas.DataFrame*) DataFrame
        """

        doy_str = kwargs.get('doy', 'doy')
        date_str = kwargs.get('date', 'date')
        u2_str = kwargs.get('u2', 'u2')
        uz_str = kwargs.get('uz', 'uz')
        z_str = kwargs.get('z', 'z')
        t_mean_str = kwargs.get('t_mean', 't_mean')
        t_min_str = kwargs.get('t_min', 't_min')
        t_max_str = kwargs.get('t_max', 't_max')
        rh_mean_str = kwargs.get('rh_mean', 'rh_mean')
        rh_min_str = kwargs.get('rh_min', 'rh_min')
        rh_max_str = kwargs.get('rh_max', 'rh_max')
        rs_str = kwargs.get('rs', 'rs')
        n_str = kwargs.get('n', 'n')
        g_str = kwargs.get('g', 'g')

        columns = df.columns
        doy = df[doy_str].values if doy_str in columns else None
        date = df[date_str].values if date_str in columns else None
        u2 = df[u2_str].values if u2_str in columns else None
        uz = df[uz_str].values if uz_str in columns else None
        z = df[z_str].values if z_str in columns else None
        t_mean = df[t_mean_str].values if t_mean_str in columns else None
        t_min = df[t_min_str].values if t_min_str in columns else None
        t_max = df[t_max_str].values if t_max_str in columns else None
        rh_mean = df[rh_mean_str].values if rh_mean_str in columns else None
        rh_min = df[rh_min_str].values if rh_min_str in columns else None
        rh_max = df[rh_max_str].values if rh_max_str in columns else None
        rs = df[rs_str].values if rs_str in columns else None
        n = df[n_str].values if n_str in columns else None
        g = df[g_str].values if g_str in columns else None

        self.et0(doy=doy, date=date, u2=u2, uz=uz, z=z, t_mean=t_mean, t_min=t_min, t_max=t_max,
                 rh_mean=rh_mean, rh_min=rh_min, rh_max=rh_max, rs=rs, n=n, g=g)

        show_all = kwargs.get('show_all', True)
        if show_all:
            if doy is None:
                df['DoY'] = self.doy
            df['Lambda'] = self.ld
            df['Psy'] = self.pc
            df['Delta'] = self.s
            df['es'] = self.es
            df['ea'] = self.ea
            df['Rs'] = self.rs
            df['Rns'] = self.rns
            df['Rnl'] = self.rnl
            df['ET0r'] = self.etr
            df['ET0w'] = self.etw
            df['ET0'] = self.et
        else:
            if kwargs.get('Lambda', False):
                df['Lambda'] = self.ld
            if kwargs.get('Psy', False):
                df['Psy'] = self.pc
            if kwargs.get('Delta', False):
                df['Delta'] = self.s
            if kwargs.get('es', False):
                df['es'] = self.es
            if kwargs.get('ea', False):
                df['ea'] = self.ea
            if kwargs.get('Rs', False):
                df['Rs'] = self.rs
            if kwargs.get('Rns', False):
                df['Rns'] = self.rns
            if kwargs.get('Rnl', False):
                df['Rnl'] = self.rnl
            if kwargs.get('ET0r', False):
                df['ET0r'] = self.etr
            if kwargs.get('ET0w', False):
                df['ET0w'] = self.etw
            if kwargs.get('ET0', True):
                df['ET0'] = self.et
        return df


