import math
import numpy as np
import pandas as pd


class PenmanMonteithDaily(object):
    r"""The class *PenmanMonteithDaily* calculates daily potential evapotranspiration according to the Penman-Monteith
    equation as described in `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ (Allen et al.,
    1998). By using the default parameters, the outcome results in evapotranspiration values for a hypothetical grass
    reference crop (:math:`h=12` *cm*; :math:`albedo=0.23`, and :math:`LAI=2.88`). The default values also assume 2 meters height
    of wind and humidity observations as well as soil heat flux density :math:`G=0.0` *MJ/m²day*. Default values can be changed in the keyword
    arguments (\*\*kwargs).

    The class *PenmanMonteithDaily* solves equation 3 in
    `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_:

    .. math::
       ET = \frac{\Delta (R_n - G) + \rho_a c_p \frac{e_s - e_a}{r_a}}
       {\lambda \left[ \Delta + \gamma \left( 1 + \frac{r_s}{r_a} \right) \right]}
       \tag{eq. 3, p. 19}


    :param elevation: elevation above sea level *[m]*. Used in :meth:`clear_sky_shortwave_radiation_daily` and
        :meth:`atmospheric_pressure`
    :type elevation: float
    :param latitude: latitude in decimal degrees. Used in :meth:`sunset_hour_angle` and :meth:`extraterrestrial_radiation`
    :type latitude: float

    :Keyword Arguments:

       * **albedo** (*float*) - albedo or canopy reflection coefficient (:math:`\alpha` *[-]*). Range: :math:`0.0  \leq \alpha \leq 1.0`. Default value: :math:`albedo=0.23` for the hypothetical grass reference crop. Used in :meth:`net_shortwave_radiation`
       * **h** (*float*) - crop height *[m]*. Default value: :math:`h=0.12` for the hypothetical grass reference crop. Required to calculate the zero plane displacement height (:math:`d` *[m]*) and the roughness length governing momentum (:math:`z_{om}` *[m]*), both necessary for the aerodynamic resistance (:math:`r_a` *[s/m]*). See :meth:`aerodynamic_resistance_factor`
       * **lai** (*float*) - leaf area index (:math:`LAI` *[-]*). Default value: :math:`lai=2.88` for the hypothetical grass reference crop (see *BOX 5* in `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_). See :meth:`bulk_surface_resistance`
       * **rl** (*float*) - bulk stomatal resistance of well-illuminated leaf (:math:`r_l` *[s/m]*). Default value: :math:`rl=100.0` for any crop. See :meth:`bulk_surface_resistance`
       * **zm** (*float*) - height of wind measurements *[m]*. Default value: :math:`zm=2.0`. Required to calculate aerodynamic resistance (:math:`r_a` *[s/m]*). See :meth:`aerodynamic_resistance_factor`
       * **zh** (*float*) - height of humidity measurements *[m]*. Default value: :math:`zh=2.0`. Required to calculate aerodynamic resistance (:math:`r_a` *[s/m]*). See :meth:`aerodynamic_resistance_factor`.
       * **g** (*float*) - soil heat flux density (:math:`G` *[MJ/m²day]*). Default value: :math:`g=0.0`. This corresponds to :math:`G` in eq. 3, p. 19 above. It can be also given in daily time steps in :meth:`et0`

    .. note::
        Only :attr:`elevation` and :attr:`latitude` are mandatory parameters. :attr:`albedo`, :attr:`h`, and :attr:`lai` should be
        given only when calculating evapotranspiration for crops other than reference grass.

    :ivar epsilon: ratio molecular weight of water vapour/dry air (:math:`\varepsilon` *[-]*). Variable value: :math:`epsilon = 0.622`
    :ivar r: specific gas constant *[kJ/kg.K]*. Variable value: :math:`r = 0.287`
    :ivar karman: von Karman constant (:math:`k` *[-]*), see FAO 56 eq. 4. Variable value: :math:`karman=0.41`
    :ivar karman: von Karman constant (:math:`k` *[-]*), see `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ eq. 4. Variable value: :math:`karman=0.41`
    :ivar d_factor: factor of the zero plane displacement height (:math:`d`) *[-]*. Variable value: :math:`d\_factor = 2.0 / 3.0`
    :ivar zom_factor: factor of the roughness length governing momentum transfer (:math:`z_{om}`) *[-]*. Variable value: :math:`zom\_factor = 0.123`
    :ivar zoh_factor: factor of the roughness length governing transfer of heat and vapour (:math:`z_{oh}`) *[-]*. Variable value: :math:`zoh\_factor = 0.1`
    :ivar lai_active_factor: factor of the active (sunlit) leaf area index :math:`LAI_{active}` *[-]* (it considers that generally only the upper half of dense clipped grass is actively contributing to the surface heat and vapour transfer). Variable value: :math:`lai\_active\_factor = 0.5`

    The calculation consists of creating an instance of :class:`PenmanMonteithDaily` and calling the method
    :meth:`et0`::

        - pm = PenmanMonteithDaily(elevation, latitude, ...)
        - et0 = pm.et0(...)

    or creating an instance of :class:`PenmanMonteithDaily` and calling the method :meth:`et0_frame` given a
    *pandas.DataFrame()* as input parameter::

        - pm = PenmanMonteithDaily(elevation, latitude, ...)
        - df = pm.et0_frame(df, ...)

    The following are object attributes:

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
        self.lamda = None
        self.delta = None
        self.psych = None
        self.daylight_hours = None
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

        self.epsilon = 0.622
        self.r = 0.287
        self.karman = 0.41
        self.d_factor = 2.0 / 3.0
        self.zom_factor = 0.123
        self.zoh_factor = 0.1
        self.lai_active_factor = 0.5

        if latitude:
            days = np.array(range(367))
            latitude = np.radians(latitude)
            dr_366 = self.inverse_relative_distance_earth_sun(days)
            sd_366 = np.array([self.solar_declination(day) for day in range(367)])
            ws_366 = np.array([self.sunset_hour_angle(latitude, s) for s in sd_366])
            self.daylight_hours_366 = np.array([PenmanMonteithDaily.daylight_hours(w) for w in ws_366])
            self.ra_366 = np.array([self.extraterrestrial_radiation(
                dr_366[i], ws_366[i], latitude, sd_366[i]) for i in range(len(dr_366))])
            self.rs0_366 = np.array([self.clear_sky_shortwave_radiation_daily(
                ra, elevation=elevation) for ra in self.ra_366])
        else:
            self.daylight_hours_366 = None
            self.ra_366 = None
            self.rs0_366 = None

        self.elevation = elevation
        """elevation in meters above sea level *[m]*"""

        self.p = atmospheric_pressure(self.elevation)
        """atmospheric pressure *[kPa]*"""

        ra_factor = self.aerodynamic_resistance_factor()

        self.f1 = 86400 * self.epsilon / (1.01 * self.r * ra_factor)
        """f1 = (specific heat at constant pressure) * (mean air density at constant pressure) /
             (1.01 * :attr:`r` * :meth:`aerodynamic_resistance_factor`). `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ Box 6"""
        self.f2 = self.bulk_surface_resistance() / ra_factor
        r""":math:`f_1 = \frac{rs}{f_{ra}}` with :math:`f_{ra}` = :meth:`aerodynamic_resistance_factor`"""

    def reset(self):
        r"""Reset the following output attributes before calculating :math:`ETo`: :math:`doy`, :math:`u2`, :math:`lamda`, :math:`delta`, :math:`psych, :math:`daylight_hours`,
        :math:`es`, :math:`ea`, :math:`ra`, :math:`rs`, :math:`rs0`, :math:`rns`, :math:`rnl`, :math:`rn`, :math:`etr`, :math:`etw`, and :math:`et`
        """
        self.doy = None
        self.u2 = None
        self.lamda = None
        self.delta = None
        self.psych = None
        self.daylight_hours = None
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

    def aerodynamic_resistance_factor(self):
        r"""The aerodynamic resistance :math:`r_a` is by (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 4, p. 20):

        .. math::

            r_a = \frac{ \ln \left( \frac{z_m - d}{z_{om}} \right) \ln \left( \frac{z_h - d}{z_{oh}} \right) } { k^2 u_z }

        where :math:`u_z` is the wind speed at height :math:`z` (see :meth:`et0`) and :math:`k` is the von Karman's
        constant defined in :attr:`karman`. :math:`zm` and :math:`zh` are crop dependent optional parameters
        in :class:`PenmanMonteithDaily` with default values for reference grass.

        The aerodynamic resistance factor :math:`f_{r_a}` is constant for a given crop:

        .. math::

            f_{r_a} = \frac{ \ln \left( \frac{z_m - d}{z_{om}} \right) \ln \left( \frac{z_h - d}{z_{oh}} \right) } { k^2}

        with the zero plane displacement height (:math:`d`):

        .. math::

            d = f_d * h

        and roughness length governing momentum transfer (:math:`z_{om}`):

        .. math::

            z_{om} = f_{zom} * h

        where :math:`f_d` is defined in :attr:`d_factor` and :math:`f_{zom}` in :attr:`zom_factor`

        The equation is restricted for neutral stability conditions, i.e., where temperature, atmospheric pressure, and
        wind velocity distributions follow nearly adiabatic conditions (no heat exchange). ... However, when predicting
        ETo in the well- watered reference surface, heat exchanged is small, and therefore stability correction is
        normally not required.

        Many studies have explored the nature of the wind regime in
        plant canopies. Zero displacement heights and roughness lengths have to be considered when the surface is
        covered by vegetation. The factors depend upon the crop height and architecture. Several empirical equations
        for the estimate of d, zom and zoh have been developed (FAO 56, p. 20).

        :return: (*float*) aerodynamic resistance factor :math:`f_{r_a}`
        """

        # zero plane displacement height, d [m]
        d = self.d_factor * self.h

        # roughness length governing momentum transfer [m]
        zom = self.zom_factor * self.h

        # roughness length governing transfer of heat and vapour [m]
        zoh = self.zoh_factor * zom

        return math.log((self.zm - d) / zom) * math.log((self.zh - d) / zoh) / (self.karman ** 2)

    def bulk_surface_resistance(self):
        r"""
        .. math::

            r_s = \frac{ r_l } { LAI_{active} }
            \tag{eq. 5, p. 21}

        where

            * :math:`r_l` (*float*): bulk stomatal resistance of the well-illuminated leaf *[s/m]*
            * :math:`LAI_{active}` (*float*): active (sunlit) leaf area index *[m² (leaf area) / m² (soil surface)]*

        A general equation for :math:`LAI_{active}` is:

        .. math::

            LAI_{active} = 0.5 LAI

        with

        .. math::

            LAI = 24 h

        where :math:`h` is an optional input parameter in :class:`PenmanMonteithDaily`

        :return: (*float*): :math:`r_s` - (bulk) surface resistance *[s/m]*
        """
        #
        # active (sunlit) leaf area index [m^2 (leaf area) / m^2 (soil surface)]
        lai_active = self.lai_active_factor * self.lai

        rs = self.rl / lai_active
        return rs

    @staticmethod
    def to_u2(uz, z):
        r""" Return the calculated wind speed at 2 meters above ground surface *[m/s]*:

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
        r"""Return the extraterrestrial radiation :math:`R_a` *[MJ/m²day]* according to the following equation
        (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 21, p. 46):

        .. math::

            R_a = \frac{24(60)}{\pi} G_{sc} d_r [ \omega_s \sin(\varphi) \sin(\delta) + \cos(\varphi) \cos(\delta)
            \sin(\omega_s)]

        :param dr: inverse relative distance Earth-Sun *[-]*
        :type dr: float
        :param ws: sunset hour angle *[rad]*
        :type ws: float
        :param lat: latitude *[rad]*
        :type lat: float
        :param sd: solar declination *[rad]*
        :type sd: float
        :return: *(float or np.array)* daily extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*
        """
        # solar_constant = 0.0820 # MJ.m-2.min-1
        # (24.0 * 60.0 / pi) * solar_constant = 37.586031360582005
        return 37.586031360582005 * dr * (ws * np.sin(lat) * np.sin(sd) + np.cos(lat) * np.cos(sd) * np.sin(ws))

    @staticmethod
    def inverse_relative_distance_earth_sun(day):
        r"""Return the inverse relative distance Earth-Sun (:math:`d_r`) according to `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 23, p. 46:

        .. math::

            d_r = 1 + 0.033 \cos{ \left( \frac{2 \pi}{365} J \right)}

        :param day: day of the year (:math:`J` *[-]*). Range: :math:`1 \leq J \leq 365`
        :type day: int
        :return: *(float or np.array)* inverse relative distance Earth-Sun *[-]*
        """
        # 2.0 * pi / 365 = 0.01721420632103996
        return 1 + 0.033 * np.cos(0.01721420632103996 * day)

    @staticmethod
    def solar_declination(day):
        r"""Return the solar declination (:math:`\delta`) according to `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 24, p. 46:

        .. math::

            \delta = 0.409 \sin{ \left( \frac{2 \pi}{365} J - 1.39\right)}

        :param day: day of the year (:math:`J` *[-]*). Range: :math:`1 \leq J \leq 365`
        :type day: int
        :return: (*float or np.array*) solar declination *[rad]*
        """
        # 2.0 * pi / 365 = 0.01721420632103996
        return 0.409 * np.sin(0.01721420632103996 * day - 1.39)

    @staticmethod
    def sunset_hour_angle(lat, sd):
        r"""Return the sunset hour angle (:math:`\omega_s`) according to `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 25, p. 46:

        .. math::

            \omega_s = \arccos{ \left[-tan(\varphi)tan(\delta)\right]}

        :param lat: latitude (:math:`\varphi`) *[rad]*
        :type lat: float or np.array
        :param sd: solar declination (:math:`\delta`) *[rad]*
        :type sd: float or np.array
        :return: (*float or np.array*) sunset hour angle *[rad]*
        """
        return np.arccos(-np.tan(sd) * np.tan(lat))

    @staticmethod
    def daylight_hours(ws):
        r"""Return the daylight hours (:math:`N`) given by (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 34, p. 49):

        .. math::

            N = \frac{24}{\pi} \omega_s

        where :math:`\omega_s` is the sunset hour angle in radians given :meth:`sunset_hour_angle`

        :param ws: sunset hour angle (:math:`\omega_s`) *[rad]*
        :type ws: float or np.numpy
        :return: (*float or np.numpy*) daylight hours *[hour]*
        """
        # 24.0 / pi = 7.639437268410976
        return 7.639437268410976 * ws

    @staticmethod
    def clear_sky_shortwave_radiation_daily(ra, elevation=0.0, a=0.25, b=0.50):
        r"""Return the clear-sky shortwave radiation (:math:`R_{so}`), when :math:`n = N`. :math:`R_{so}` is required for computing :meth:`net_longwave_radiation`.

        For near sea level or when calibrated values for :math:`as` and :math:`bs` are available (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 36, p. 51):

        .. math::

           R_{so} = (a_s + b_s ) R_a


        where:

          :math:`R_{so}` --- clear-sky solar radiation *[MJ/m²day]*

          :math:`as+bs` --- fraction of extraterrestrial radiation reaching the earth on clear-sky days (:math:`n = N`)

        When calibrated values for :math:`as` and :math:`bs` are not available (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 37, p. 51):

        .. math::

           R_{so} = (0.75 + 2 * 10^−5 z) R_a

        where :math:`z` station elevation above sea level *[m]*

        :param ra: extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*
        :type ra: float or np.numpy
        :param elevation: see :attr:`elevation`
        :type elevation: float or np.numpy
        :param a: regression constant (:math:`as` *[-]*). Default value: :math:`a=0.25`. It expresses the fraction of extraterrestrial radiation reaching the earth on overcast days (:math:`n = 0`)
        :type a: float or np.numpy
        :param b: regression constant (:math:`bs` *[-]*). Default value: :math:`b=0.50`. The expression :math:`a+b` indicates the fraction of extraterrestrial radiation reaching the earth on clear days (:math:`n = N`)
        :type b: float or np.numpy
        :return: (*float or np.numpy*) daily clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*
        """
        rs0 = ((a + b) + 2e-5 * elevation) * ra
        return rs0

    @staticmethod
    def shortwave_radiation(ra, n, nt, a_s=0.25, b_s=0.50):
        r"""Return the daily shortwave radiation in *[MJ/m²day]* according to the Angstrom formula (see e.g., `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 35, p. 50):

        .. math::

           R_s = \left( a_s + b_s \frac{n}{N} \right) R_a

        Depending on atmospheric conditions (humidity, dust) and solar declination (latitude and month), the Angstrom
        values :math:`a_s` and :math:`b_s` will vary. Where no actual solar radiation data are available and no
        calibration has been carried out for improved :math:`a_s` and :math:`b_s` parameters, the values
        :math:`a_s = 0.25` and :math:`b_s = 0.50` are recommended.

        :param ra: extraterrestrial radiation (:math:`R_a`) *[MJ/m²day]*
        :type ra: float or np.array
        :param n: actual duration of sunshine or cloudless hours (:math:`n`) *[hour]*
        :type n: float or np.array
        :param nt: maximum possible duration of sunshine or daylight hours (:math:`N`) *[hour]*
        :type nt: float, np.array
        :param a_s: regression constant (:math:`as` *[-]*). Default value: :math:`a_s=0.25`. It expresses the fraction of extraterrestrial radiation reaching the earth on overcast days (:math:`n = 0`)
        :type a_s: float or np.numpy
        :param b_s: regression constant (:math:`bs` *[-]*). Default value: :math:`b_s=0.50`. The expression :math:`a_s+b_s` indicates the fraction of extraterrestrial radiation reaching the earth on clear days (:math:`n = nt`)
        :type b_s: float or np.numpy
        :return: (*float, np.array*) daily total shortwave radiation (:math:`R_s`) reaching the earth *[MJ/m²day]*

        .. note::
                If shortwave radiation (i.e., solar radiation) measurements are available, :meth:`shortwave_radiation` function
                is no needed. Measurements of shortwave radiation may be directly used as input data in %mp.

        """
        rns = (a_s + b_s * n / nt) * ra
        return rns


    @staticmethod
    def net_shortwave_radiation(rs, albedo):
        r"""The net shortwave radiation (:math:`R_{ns}`) resulting from the balance between incoming and reflected solar radiation
        is given by (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 38, p. 51):

        .. math::

            R_{ns} = (1 − \alpha) R_s

        :param rs: daily shortwave radiation (:math:`R_s`) *[MJ/m²day]*
        :type rs: float or np.array
        :param albedo: albedo or reflection coefficient (:math:`\alpha` *[-]*). Range: :math:`0.0  \leq \alpha \leq 1.0` (:math:`albedo=0.23` for the hypothetical grass reference crop). See :class:`PenmanMonteithDaily` and :meth:`et0`
        :type albedo: float or np.array
        :return: (*float or np.array*) daily net shortwave radiation (:math:`R_{ns}`) reaching the earth *[MJ/m²day]*
        """
        return (1.0 - albedo) * rs

    @staticmethod
    def net_longwave_radiation(t_min, t_max, rs, rs0, ea=None):
        r"""Return the net outgoing longwave radiation (:math:`R_{nl}`) *[MJ/m²day]* given by (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_, eq. 39, p. 52):

        .. math::

            R_{nl} = \sigma\left[\frac{T_{max,K}^4 + T_{min,K}^4}{2}\right](0.34-0.14\sqrt{e_a})\left(1.35\frac{R_s}{R_{so}}-0.35\right)

        :param t_min: minimum temperature during the 24-hour period (:math:`T_{max}`) *[°C]*
        :type t_min: float or np.array
        :param t_max: maximum temperature during the 24-hour period (:math:`T_{min}`) *[°C]*
        :type t_max: float or np.array
        :param rs: measured or calculated shortwave radiation (:math:`R_s`) *[MJ/m²day]*
        :type rs: float or np.array
        :param rs0: calculated clear-sky shortwave radiation (:math:`R_{so}`) *[MJ/m²day]*
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
            rln = 4.903e-9 * (t_min ** 4 + t_max ** 4) * 0.5 * (-0.02 + 0.261 * np.exp(-7.77e10 ** -4 * t_mean ** 2)) * \
                  (1.35 * rs / rs0 - 0.35)
        return rln

    def et0(self, **kwargs):
        r"""Returns potential evapotranspiration :math:`ETo` in *[mm/day]* according to the procedures described in `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_.

        - Reference (grass) potencial evapotranspiration is returned for default constructor values.

        - If Kwargs are arrays, their lengths must be the same.

        :Keyword Arguments:

           * **date** (str, datetime.date, datetime.datetime, pandas.TimeStamp, or np.array)
           * **doy** (*int or np.array*) - day of the year (:math:`J` *[-]*). Range: :math:`1 \leq J \leq 365`. It is not used if date is given
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
           * **g** (*float or np.array*) - soil heat flux density *[MJ/m²day]*
           * **a_s** (*float or np.array*) - see :meth:`shortwave_radiation`. Default :math:`a_s = 0.25`
           * **b_s** (*float or np.array*) - see :meth:`shortwave_radiation`. Default :math:`b_s = 0.50`
        :return: (*float or np.array*) potential evapotranspiration *[mm/day]*

        Example 1::

            >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
            >>> et0 = pm.et0(doy=187, u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, n=9.25)
            >>> print(et0)
            3.872968723753793

        Example 2::

            >>> pm = PenmanMonteithDaily(elevation=100, latitude=50.80)
            >>> et0 = pm.et0(date='2001-07-06', u2=2.078, t_min=12.3, t_max=21.5, rh_min=63, rh_max=84, n=9.25)
            >>> print(et0)
            3.872968723753793

        Example 3::

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
        b_s = kwargs.get('a_s', 0.50)

        if t_mean is None:
            t_mean = (t_min + t_max) / 2.0

        self.lamda = latent_heat_of_vaporization(t_mean)
        # In FAO 56, where delta occurs in the numerator and denominator, the slope
        # of the vapour pressure curve is calculated using mean air temperature (Equation 9)
        self.delta = slope_of_saturation_vapour_pressure_curve(t_mean)
        self.psych = psychrometric_constant(self.p)

        self.es = saturation_vapour_pressure(t_min, t_max)
        self.ea = actual_vapour_pressure(rh_min=rh_min, rh_max=rh_max, t_min=t_min, t_max=t_max)

        try:
            self.ra = np.array([self.ra_366[i] for i in self.doy])
            self.rs0 = np.array([self.rs0_366[i] for i in self.doy])
            if self.rs is None:
                self.daylight_hours = np.array([self.daylight_hours_366[i] for i in self.doy])
                self.rs = self.shortwave_radiation(self.ra, n, self.daylight_hours, a_s, b_s)
                # FAO56 eq. 39. The Rs/Rso term in equation 39 must be limited so that Rs/Rso ≤ 1.0.
                self.rs = np.where(self.rs > self.rs0, self.rs0, self.rs)
        except TypeError:
            self.ra = self.ra_366[self.doy]
            self.rs0 = self.rs0_366[self.doy]
            if self.rs is None:
                self.daylight_hours = self.daylight_hours_366[self.doy]
                self.rs = self.shortwave_radiation(self.ra, n, self.daylight_hours, a_s, b_s)
                # FAO56 eq. 39. The Rs/Rso term in equation 39 must be limited so that Rs/Rso ≤ 1.0.
                self.rs = self.rs0 if self.rs > self.rs0 else self.rs

        self.rns = self.net_shortwave_radiation(self.rs, self.albedo)
        self.rnl = self.net_longwave_radiation(t_min, t_max, self.rs, self.rs0, self.ea)
        self.rn = self.rns - self.rnl

        # denominator of FAO 56 eq. 3
        etd = self.lamda * (self.delta + self.psych * (1 + self.f2 * self.u2))

        # ETo energy component of FAO 56 eq. 3
        self.etr = self.delta * (self.rn - g) / etd
        # ETo wind component of FAO 56 eq. 3
        self.etw = (self.lamda * self.psych * self.u2 * self.f1 * (self.es - self.ea) / (t_mean + 273.0)) / etd
        # Reference evapotranspiration
        self.et = self.etr + self.etw
        self.et = np.where(self.et < 0.0, 0.0, self.et)
        try:
            self.et = float(self.et)
        except TypeError:
            pass
        return self.et

    def et0_frame(self, df, **kwargs):
        """

        :param df:
        :type df: pandas.DataFrame

        :Keyword Arguments:

           * **date** (str, datetime.date, datetime.datetime, pandas.TimeStamp, numpy.array):
           * **doy** (): a
           * **date** (): a
           * **u2** (): a
           * **uz** (): a
           * **z** (): a
           * **t_mean** (): a
           * **t_min** (): a
           * **t_max** (): a
           * **rh_mean** (): a
           * **rh_min** (): a
           * **rh_max** (): a
           * **rs** (): a
           * **n** (): a
           * **g** (): a
           * **Lamda** (): default False.
           * **Psy** (): default False.
           * **Delta** (): default False.
           * **es** (): default False.
           * **ea** (): default False.
           * **Rs** (): default False.
           * **Rns** (): default False.
           * **Rnl** (): default False.
           * **ET0r** (): default False.
           * **ET0w** (): default False.
           * **ET0** (): default True.

        :return: df
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
            df['Lambda'] = self.lamda
            df['Psy'] = self.psych
            df['Delta'] = self.delta
            df['es'] = self.es
            df['ea'] = self.ea
            df['Rs'] = self.rs
            df['Rns'] = self.rns
            df['Rnl'] = self.rnl
            df['ET0r'] = self.etr
            df['ET0w'] = self.etw
            df['ET0'] = self.et
        else:
            if kwargs.get('Lamda', False):
                df['Lambda'] = self.lamda
            if kwargs.get('Psy', False):
                df['Psy'] = self.psych
            if kwargs.get('Delta', False):
                df['Delta'] = self.delta
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


def atmospheric_pressure(z):
    """ Calculates the atmospheric pressure in *[kPa]* as a function of the elevation above sea level.

    `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ eq. 7

    The atmospheric pressure, P, is the pressure exerted by the weight of the earth's atmosphere. Evaporation at high
    altitudes is promoted due to low atmospheric pressure as expressed in the psychrometric constant. The effect is,
    however, small and in the calculation procedures, the average value for a LOCATION is sufficient. A simplification
    of the ideal gas law, assuming 20 *°C* for a standard atmosphere, can be employed to calculate :math:`P` (`FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_)

    :param z: elevation above sea level *[m]*
    :type z:
    :return: (*???*) atmospheric pressure *[kPa]*
    """
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0)**5.26


def latent_heat_of_vaporization(temperature=20):
    """Return the Latent Heat of Vaporization in MJ/kg

    `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ eq. 3-1

    :param temperature: air temperature in *[°C]*. Default value: :math:`temperature=20`
    :type temperature:
    :return: (*???*) Default = :math:`2.45378` *MJ/kg*
    """
    return 2.501 - 2.361e-3 * temperature


def psychrometric_constant(p, a_psy=0.000665):
    """Return the psychrometric constant in *kPa/°C*

    :param p: atmospheric pressure *[kPa]*
    :type p:
    :param a_psy: coefficient depending on the type of the ventilation of the bulb *[1/°C]*
        Examples:
        a_psy = 0.000665 (default)
        a_psy = 0.000662 for ventilated (Asmann type) psychrometers, with an air movement of some 5 m/s
        a_psy = 0.000800 for natural ventilated psychrometers (about 1 m/s)
        a_psy = 0.001200 for non-ventilated psychrometers installed indoors
    :return: (*???*) psychrometric constant *[kPa/°C]*
    """
    return a_psy * p


def saturation_vapour_pressure(*temperature):
    """r Return `FAO 56 <http://www.fao.org/tempref/SD/Reserved/Agromet/PET/FAO_Irrigation_Drainage_Paper_56.pdf>`_ eq. 11
    :param temperature: air temperature *[°C]*
    :type temperature:
    :return: (*???*) saturation vapour pressure *[kPa]*
    """
    t = np.array([0.6108 * np.exp((17.27 * t) / (t + 237.3)) for t in temperature])
    t = np.mean(t, axis=0)
    return t


def slope_of_saturation_vapour_pressure_curve(*temperature):
    """

    :param temperature: temperature: air temperature *[°C]*
    :type temperature:
    :return: (*???*) slope of...
    """
    """
    page 37
    Inputs
        tmax :  maximum air temperature [C]
        tmin :  minimum air temperature [C]
    """
    sl = np.array([(4098.0 * saturation_vapour_pressure(t)) / ((t + 237.3) ** 2) for t in temperature])
    return np.mean(sl, axis=0)


def actual_vapour_pressure(**kwargs):
    """
    page 37 , 38 , 39

    :param kwargs:
        rh_min:  0.0 to 100.0 [%]
        rh_max:  0.0 to 100.0 [%]
        es_min:  [kPa]
        es_max:  [kPa]
        t_min: [°C]
        t_max: [°C]

        t_dew :  dewpoint temperature [°C]

        t_wet :  wet bulb temperature temperature [°C]
        t_dry :  dry bulb temperature temperature [°C]
        apsy :  coefficient depending on the type of ventilation of the wet bulb
    :return:
    :rtype:
    """
    try:
        t_min = kwargs['t_min']
        t_max = kwargs['t_max']
        rh_min = kwargs['rh_min'] / 100
        rh_max = kwargs['rh_max'] / 100
        es_min = kwargs.get('es_min', saturation_vapour_pressure(t_min))
        es_max = kwargs.get('es_max', saturation_vapour_pressure(t_max))

        return (rh_max * es_min + rh_min * es_max) / 2.0
    except KeyError:
        t_dew = kwargs.get('t_dew', None)
        return 0.6108 * math.exp((17.27 * t_dew)/(t_dew + 237.3))



    # @staticmethod
    # def net_radiation_daily(lat, t_min, t_max, albedo, day, cloudless_hours=None, ea=None):
    #     """
    #     :param lat: latitude [rad]
    #     :param t_min: minimum temperature during the 24-hour period [C]
    #     :param t_max: maximum temperature during the 24-hour period [C]
    #     :param albedo: reflection coefficient (0 <= albedo <= 1), which is 0.23 for the hypothetical grass reference crop [-]
    #     :param day: day of the year (1 to 365)
    #     :param cloudless_hours: actual duration of sunshine (cloudless hours) [hour]
    #     :param ea: actual vapour pressure [kPa]
    #     :return: daily net radiation
    #     """
    #     dr = PenmanMonteithDaily.inverse_relative_distance_earth_sun(day)
    #     sd = PenmanMonteithDaily.solar_declination(day)
    #     ws = PenmanMonteithDaily.sunset_hour_angle(lat, sd)
    #     nt = PenmanMonteithDaily.daylight_hours(ws)
    #     r0 = PenmanMonteithDaily.extraterrestrial_radiation_daily(dr, ws, lat, sd)
    #     rs = PenmanMonteithDaily.shortwave_radiation_daily(r0, cloudless_hours, nt)
    #     rs0 = PenmanMonteithDaily.shortwave_radiation_daily(r0, nt, nt)
    #     rns = PenmanMonteithDaily.net_shortwave_radiation_daily(rs, albedo)
    #     rnl = PenmanMonteithDaily.net_longwave_radiation_daily(t_min, t_max, rs, rs0, ea)
    #     return rns - rnl

