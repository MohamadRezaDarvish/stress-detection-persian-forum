from src.scraping import detect_gender_from_profile_html

def test_female_is_woman_marker():
    html = '<div><i class="is-woman"></i><span>زن</span></div>'
    assert detect_gender_from_profile_html(html)[:2] == ("زن", 1)

def test_male_is_user_marker():
    html = '<div><i class="is-user"></i><span>مرد</span></div>'
    assert detect_gender_from_profile_html(html)[:2] == ("مرد", 0)

def test_legacy_markers():
    assert detect_gender_from_profile_html(
        '<i class="iconwoman"></i><span>زن</span>'
    )[:2] == ("زن", 1)
    assert detect_gender_from_profile_html(
        '<i class="iconuser-01"></i><span>مرد</span>'
    )[:2] == ("مرد", 0)
    assert detect_gender_from_profile_html(
        '<i class="iconman"></i><span>مرد</span>'
    )[:2] == ("مرد", 0)

def test_unknown_is_not_forced_to_male_or_female():
    html = '<div class="profile"><span>اطلاعات کاربر</span></div>'
    assert detect_gender_from_profile_html(html)[:2] == ("unknown", -1)
